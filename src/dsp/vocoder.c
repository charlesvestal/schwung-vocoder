/*
 * Vocoder Audio FX Plugin
 *
 * Classic channel vocoder: analyzes the spectral envelope of a modulator signal
 * (mic/line-in from hardware input buffer) and applies it to a carrier signal
 * (synth output from the signal chain).
 *
 * Each band uses a 2nd-order state-variable bandpass filter and a single-pole
 * envelope follower with separate attack/release coefficients.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "audio_fx_api_v1.h"

/* Audio FX API v2 - instance-based */
#define AUDIO_FX_API_VERSION_2 2
#define AUDIO_FX_INIT_V2_SYMBOL "move_audio_fx_init_v2"

typedef struct audio_fx_api_v2 {
    uint32_t api_version;
    void* (*create_instance)(const char *module_dir, const char *config_json);
    void (*destroy_instance)(void *instance);
    void (*process_block)(void *instance, int16_t *audio_inout, int frames);
    void (*set_param)(void *instance, const char *key, const char *val);
    int (*get_param)(void *instance, const char *key, char *buf, int buf_len);
} audio_fx_api_v2_t;

typedef audio_fx_api_v2_t* (*audio_fx_init_v2_fn)(const host_api_v1_t *host);

#define SAMPLE_RATE 44100
#define MAX_BANDS 32

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── State-variable bandpass filter (2nd-order) ──────────────────────── */

typedef struct {
    float low;   /* lowpass state */
    float band;  /* bandpass state */
} svf_state_t;

static inline float svf_bandpass(svf_state_t *s, float input, float f, float q) {
    /* f = 2 * sin(pi * fc / sr), q = 1/Q */
    s->low  += f * s->band;
    float high = input - s->low - q * s->band;
    s->band += f * high;
    return s->band;
}

/* ── Envelope follower (single-pole, separate attack/release) ──────── */

typedef struct {
    float level;
} env_state_t;

static inline float env_follow(env_state_t *e, float input, float att, float rel) {
    float rect = fabsf(input);
    float coeff = (rect > e->level) ? att : rel;
    e->level += coeff * (rect - e->level);
    return e->level;
}

/* ── Vocoder instance ────────────────────────────────────────────────── */

typedef struct {
    /* Parameters */
    int    bands;         /* 8, 16, 24, or 32 */
    float  freq_low;      /* Hz */
    float  freq_high;     /* Hz */
    float  attack_ms;     /* ms */
    float  release_ms;    /* ms */
    float  mod_gain;      /* 0..6 modulator input gain */
    float  output_gain;   /* 0..6 output gain */
    float  mix;           /* 0..1 wet/dry */
    float  carrier_mix;   /* 0..1 noise for unvoiced */

    /* Derived per-band coefficients */
    float  band_f[MAX_BANDS];  /* SVF frequency coeff */
    float  band_q[MAX_BANDS];  /* SVF reciprocal-Q */
    float  att_coeff;          /* envelope attack */
    float  rel_coeff;          /* envelope release */

    /* Filter states (modulator + carrier, stereo) */
    svf_state_t mod_svf_l[MAX_BANDS];
    svf_state_t mod_svf_r[MAX_BANDS];
    svf_state_t car_svf_l[MAX_BANDS];
    svf_state_t car_svf_r[MAX_BANDS];
    env_state_t mod_env_l[MAX_BANDS];
    env_state_t mod_env_r[MAX_BANDS];

    /* Simple noise state for unvoiced */
    uint32_t noise_seed;
} vocoder_instance_t;

static const host_api_v1_t *g_host = NULL;

/* ── Helpers ─────────────────────────────────────────────────────────── */

static void voc_log(const char *msg) {
    if (g_host && g_host->log) {
        char buf[256];
        snprintf(buf, sizeof(buf), "[vocoder] %s", msg);
        g_host->log(buf);
    }
}

/* Simple fast white noise */
static inline float noise_sample(uint32_t *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return (float)(int32_t)(*seed) / 2147483648.0f;
}

/* Recalculate per-band coefficients from current parameters */
static void recalc_bands(vocoder_instance_t *v) {
    float log_low  = logf(v->freq_low);
    float log_high = logf(v->freq_high);
    int n = v->bands;

    for (int i = 0; i < n; i++) {
        /* Logarithmically spaced center frequencies */
        float t = (n > 1) ? (float)i / (float)(n - 1) : 0.5f;
        float fc = expf(log_low + t * (log_high - log_low));

        /* SVF frequency coefficient: 2 * sin(pi * fc / sr) */
        float f = 2.0f * sinf((float)M_PI * fc / (float)SAMPLE_RATE);
        /* Clamp to avoid instability */
        if (f > 1.0f) f = 1.0f;
        v->band_f[i] = f;

        /* Q proportional to band spacing — wider bands at low count */
        /* Q ~ sqrt(n) gives decent overlap */
        float Q = 1.0f + 0.5f * sqrtf((float)n);
        v->band_q[i] = 1.0f / Q;
    }

    /* Envelope coefficients from time constants */
    float att_ms = v->attack_ms;
    float rel_ms = v->release_ms;
    if (att_ms < 0.1f) att_ms = 0.1f;
    if (rel_ms < 0.1f) rel_ms = 0.1f;
    v->att_coeff = 1.0f - expf(-1.0f / (att_ms * 0.001f * (float)SAMPLE_RATE));
    v->rel_coeff = 1.0f - expf(-1.0f / (rel_ms * 0.001f * (float)SAMPLE_RATE));
}

/* Clear all filter states */
static void clear_filters(vocoder_instance_t *v) {
    memset(v->mod_svf_l, 0, sizeof(v->mod_svf_l));
    memset(v->mod_svf_r, 0, sizeof(v->mod_svf_r));
    memset(v->car_svf_l, 0, sizeof(v->car_svf_l));
    memset(v->car_svf_r, 0, sizeof(v->car_svf_r));
    memset(v->mod_env_l, 0, sizeof(v->mod_env_l));
    memset(v->mod_env_r, 0, sizeof(v->mod_env_r));
}

/* Simple JSON float extraction */
static int json_get_float(const char *json, const char *key, float *out) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    while (*p == ' ' || *p == '\t') p++;
    *out = (float)atof(p);
    return 0;
}

static int json_get_int(const char *json, const char *key, int *out) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    while (*p == ' ' || *p == '\t') p++;
    *out = atoi(p);
    return 0;
}

/* Clamp helpers */
static inline float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline int clampi(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Snap band count to nearest valid value */
static int snap_bands(int v) {
    if (v <= 12) return 8;
    if (v <= 20) return 16;
    if (v <= 28) return 24;
    return 32;
}

/* ── V2 API ──────────────────────────────────────────────────────────── */

static void* v2_create_instance(const char *module_dir, const char *config_json) {
    (void)module_dir;
    (void)config_json;

    voc_log("Creating instance");

    vocoder_instance_t *v = (vocoder_instance_t *)calloc(1, sizeof(vocoder_instance_t));
    if (!v) {
        voc_log("Failed to allocate instance");
        return NULL;
    }

    /* Defaults */
    v->bands       = 16;
    v->freq_low    = 100.0f;
    v->freq_high   = 8000.0f;
    v->attack_ms   = 5.0f;
    v->release_ms  = 50.0f;
    v->mod_gain    = 2.0f;
    v->output_gain = 2.0f;
    v->mix         = 1.0f;
    v->carrier_mix = 0.1f;
    v->noise_seed  = 12345;

    recalc_bands(v);

    voc_log("Instance created");
    return v;
}

static void v2_destroy_instance(void *instance) {
    vocoder_instance_t *v = (vocoder_instance_t *)instance;
    if (!v) return;
    voc_log("Destroying instance");
    free(v);
}

static void v2_process_block(void *instance, int16_t *audio_inout, int frames) {
    vocoder_instance_t *v = (vocoder_instance_t *)instance;
    if (!v || !g_host) return;

    int n = v->bands;

    /* Read modulator from hardware audio input buffer */
    int16_t *mic_in = (int16_t *)(g_host->mapped_memory + g_host->audio_in_offset);

    float att = v->att_coeff;
    float rel = v->rel_coeff;
    float mod_gain = v->mod_gain;
    float out_gain = v->output_gain;
    float wet = v->mix;
    float dry = 1.0f - wet;
    float noise_mix = v->carrier_mix;

    for (int i = 0; i < frames; i++) {
        /* Convert carrier (synth output) to float */
        float car_l = audio_inout[i * 2]     / 32768.0f;
        float car_r = audio_inout[i * 2 + 1] / 32768.0f;

        /* Convert modulator (mic input) to float with gain */
        float mod_l = mic_in[i * 2]     / 32768.0f * mod_gain;
        float mod_r = mic_in[i * 2 + 1] / 32768.0f * mod_gain;

        /* Add noise to carrier for unvoiced/consonant content */
        float ns = noise_sample(&v->noise_seed);
        float car_noise_l = car_l + ns * noise_mix;
        float car_noise_r = car_r + ns * noise_mix;

        /* Accumulate vocoded output across bands */
        float out_l = 0.0f;
        float out_r = 0.0f;

        for (int b = 0; b < n; b++) {
            float f = v->band_f[b];
            float q = v->band_q[b];

            /* Filter modulator through bandpass → envelope */
            float mod_band_l = svf_bandpass(&v->mod_svf_l[b], mod_l, f, q);
            float mod_band_r = svf_bandpass(&v->mod_svf_r[b], mod_r, f, q);
            float env_l = env_follow(&v->mod_env_l[b], mod_band_l, att, rel);
            float env_r = env_follow(&v->mod_env_r[b], mod_band_r, att, rel);

            /* Filter carrier through same bandpass */
            float car_band_l = svf_bandpass(&v->car_svf_l[b], car_noise_l, f, q);
            float car_band_r = svf_bandpass(&v->car_svf_r[b], car_noise_r, f, q);

            /* Multiply carrier band by modulator envelope */
            out_l += car_band_l * env_l;
            out_r += car_band_r * env_r;
        }

        /* Scale output (more bands = more energy), apply output gain */
        float scale = 2.0f / sqrtf((float)n) * out_gain;
        out_l *= scale;
        out_r *= scale;

        /* Wet/dry mix */
        float mix_l = out_l * wet + car_l * dry;
        float mix_r = out_r * wet + car_r * dry;

        /* Clamp and write back */
        mix_l = clampf(mix_l, -1.0f, 1.0f);
        mix_r = clampf(mix_r, -1.0f, 1.0f);

        audio_inout[i * 2]     = (int16_t)(mix_l * 32767.0f);
        audio_inout[i * 2 + 1] = (int16_t)(mix_r * 32767.0f);
    }
}

static void v2_set_param(void *instance, const char *key, const char *val) {
    vocoder_instance_t *v = (vocoder_instance_t *)instance;
    if (!v) return;

    /* State restore from patch save */
    if (strcmp(key, "state") == 0) {
        int iv;
        float fv;
        if (json_get_int(val, "bands", &iv) == 0) {
            v->bands = snap_bands(clampi(iv, 8, 32));
        }
        if (json_get_float(val, "freq_low", &fv) == 0)
            v->freq_low = clampf(fv, 80.0f, 500.0f);
        if (json_get_float(val, "freq_high", &fv) == 0)
            v->freq_high = clampf(fv, 2000.0f, 12000.0f);
        if (json_get_float(val, "attack", &fv) == 0)
            v->attack_ms = clampf(fv, 0.1f, 50.0f);
        if (json_get_float(val, "release", &fv) == 0)
            v->release_ms = clampf(fv, 5.0f, 500.0f);
        if (json_get_float(val, "mod_gain", &fv) == 0)
            v->mod_gain = clampf(fv, 0.0f, 6.0f);
        if (json_get_float(val, "output_gain", &fv) == 0)
            v->output_gain = clampf(fv, 0.0f, 6.0f);
        if (json_get_float(val, "mix", &fv) == 0)
            v->mix = clampf(fv, 0.0f, 1.0f);
        if (json_get_float(val, "carrier_mix", &fv) == 0)
            v->carrier_mix = clampf(fv, 0.0f, 1.0f);

        clear_filters(v);
        recalc_bands(v);
        return;
    }

    float fv = (float)atof(val);

    if (strcmp(key, "bands") == 0) {
        int new_bands = snap_bands(clampi((int)fv, 8, 32));
        if (new_bands != v->bands) {
            v->bands = new_bands;
            clear_filters(v);
            recalc_bands(v);
        }
    } else if (strcmp(key, "freq_low") == 0) {
        v->freq_low = clampf(fv, 80.0f, 500.0f);
        recalc_bands(v);
    } else if (strcmp(key, "freq_high") == 0) {
        v->freq_high = clampf(fv, 2000.0f, 12000.0f);
        recalc_bands(v);
    } else if (strcmp(key, "attack") == 0) {
        v->attack_ms = clampf(fv, 0.1f, 50.0f);
        recalc_bands(v);
    } else if (strcmp(key, "release") == 0) {
        v->release_ms = clampf(fv, 5.0f, 500.0f);
        recalc_bands(v);
    } else if (strcmp(key, "mod_gain") == 0) {
        v->mod_gain = clampf(fv, 0.0f, 6.0f);
    } else if (strcmp(key, "output_gain") == 0) {
        v->output_gain = clampf(fv, 0.0f, 6.0f);
    } else if (strcmp(key, "mix") == 0) {
        v->mix = clampf(fv, 0.0f, 1.0f);
    } else if (strcmp(key, "carrier_mix") == 0) {
        v->carrier_mix = clampf(fv, 0.0f, 1.0f);
    }
}

static int v2_get_param(void *instance, const char *key, char *buf, int buf_len) {
    vocoder_instance_t *v = (vocoder_instance_t *)instance;
    if (!v) return -1;

    if (strcmp(key, "name") == 0) {
        return snprintf(buf, buf_len, "Vocoder");
    }

    /* Individual parameters */
    if (strcmp(key, "bands") == 0)
        return snprintf(buf, buf_len, "%d", v->bands);
    if (strcmp(key, "freq_low") == 0)
        return snprintf(buf, buf_len, "%.1f", v->freq_low);
    if (strcmp(key, "freq_high") == 0)
        return snprintf(buf, buf_len, "%.1f", v->freq_high);
    if (strcmp(key, "attack") == 0)
        return snprintf(buf, buf_len, "%.1f", v->attack_ms);
    if (strcmp(key, "release") == 0)
        return snprintf(buf, buf_len, "%.1f", v->release_ms);
    if (strcmp(key, "mod_gain") == 0)
        return snprintf(buf, buf_len, "%.2f", v->mod_gain);
    if (strcmp(key, "output_gain") == 0)
        return snprintf(buf, buf_len, "%.2f", v->output_gain);
    if (strcmp(key, "mix") == 0)
        return snprintf(buf, buf_len, "%.2f", v->mix);
    if (strcmp(key, "carrier_mix") == 0)
        return snprintf(buf, buf_len, "%.2f", v->carrier_mix);

    /* Full state for patch save/restore */
    if (strcmp(key, "state") == 0) {
        return snprintf(buf, buf_len,
            "{\"bands\":%d,\"freq_low\":%.1f,\"freq_high\":%.1f,"
            "\"attack\":%.1f,\"release\":%.1f,\"mod_gain\":%.2f,"
            "\"output_gain\":%.2f,\"mix\":%.2f,\"carrier_mix\":%.2f}",
            v->bands, v->freq_low, v->freq_high,
            v->attack_ms, v->release_ms, v->mod_gain,
            v->output_gain, v->mix, v->carrier_mix);
    }

    /* Shadow UI hierarchy */
    if (strcmp(key, "ui_hierarchy") == 0) {
        const char *hierarchy = "{"
            "\"modes\":null,"
            "\"levels\":{"
                "\"root\":{"
                    "\"children\":null,"
                    "\"knobs\":[\"mod_gain\",\"output_gain\",\"bands\",\"mix\",\"freq_low\",\"freq_high\",\"attack\",\"release\"],"
                    "\"params\":[\"mod_gain\",\"output_gain\",\"bands\",\"mix\",\"freq_low\",\"freq_high\",\"attack\",\"release\",\"carrier_mix\"]"
                "}"
            "}"
        "}";
        int len = strlen(hierarchy);
        if (len < buf_len) {
            strcpy(buf, hierarchy);
            return len;
        }
        return -1;
    }

    /* Chain params metadata for shadow parameter editor */
    if (strcmp(key, "chain_params") == 0) {
        const char *params_json = "["
            "{\"key\":\"bands\",\"name\":\"Bands\",\"type\":\"enum\",\"options\":[\"8\",\"16\",\"24\",\"32\"],\"default\":\"16\"},"
            "{\"key\":\"freq_low\",\"short_name\":\"Low\",\"name\":\"Low Freq\",\"type\":\"float\",\"min\":80,\"max\":500,\"default\":100,\"step\":10,\"unit\":\"Hz\"},"
            "{\"key\":\"freq_high\",\"short_name\":\"High\",\"name\":\"High Freq\",\"type\":\"float\",\"min\":2000,\"max\":12000,\"default\":8000,\"step\":100,\"unit\":\"Hz\"},"
            "{\"key\":\"attack\",\"name\":\"Attack\",\"type\":\"float\",\"min\":0.1,\"max\":50,\"default\":5,\"step\":0.5,\"unit\":\"ms\"},"
            "{\"key\":\"release\",\"name\":\"Release\",\"type\":\"float\",\"min\":5,\"max\":500,\"default\":50,\"step\":5,\"unit\":\"ms\"},"
            "{\"key\":\"mod_gain\",\"short_name\":\"Gain\",\"name\":\"Mod Gain\",\"type\":\"float\",\"min\":0,\"max\":6,\"default\":2,\"step\":0.1},"
            "{\"key\":\"output_gain\",\"short_name\":\"Out\",\"name\":\"Out Gain\",\"type\":\"float\",\"min\":0,\"max\":6,\"default\":2,\"step\":0.1},"
            "{\"key\":\"mix\",\"name\":\"Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"default\":1,\"step\":0.01},"
            "{\"key\":\"carrier_mix\",\"name\":\"Unvoiced\",\"type\":\"float\",\"min\":0,\"max\":1,\"default\":0.1,\"step\":0.01}"
        "]";
        int len = strlen(params_json);
        if (len < buf_len) {
            strcpy(buf, params_json);
            return len;
        }
        return -1;
    }

    return -1;
}

/* ── Entry point ─────────────────────────────────────────────────────── */

static audio_fx_api_v2_t g_fx_api_v2;

audio_fx_api_v2_t* move_audio_fx_init_v2(const host_api_v1_t *host) {
    g_host = host;

    memset(&g_fx_api_v2, 0, sizeof(g_fx_api_v2));
    g_fx_api_v2.api_version    = AUDIO_FX_API_VERSION_2;
    g_fx_api_v2.create_instance  = v2_create_instance;
    g_fx_api_v2.destroy_instance = v2_destroy_instance;
    g_fx_api_v2.process_block    = v2_process_block;
    g_fx_api_v2.set_param        = v2_set_param;
    g_fx_api_v2.get_param        = v2_get_param;

    voc_log("Vocoder v2 API initialized");
    return &g_fx_api_v2;
}
