#ifndef AVID_LAYER3_C_ABI_HARNESS_H
#define AVID_LAYER3_C_ABI_HARNESS_H

#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    char test_id[64];
    char test_name[128];
    char impl_file[256];
    char impl_func[256];
    char description[512];
    char repo_root[512];
    char validation_dir[512];
    char c_compiler[256];
    char expected_summary[512];
    char actual_summary[512];
    char notes[32][256];
    int  note_count;
    int  checks_run;
    int  checks_failed;
    struct timespec t0;
} c_abi_ctx_t;

static inline void c_abi_mkdir_p(const char *dir_path) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", dir_path);
    size_t len = strlen(tmp);
    if (len == 0) return;
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';
    for (char *p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static inline void c_abi_init(c_abi_ctx_t *ctx, int argc, char **argv,
                              const char *default_id, const char *name,
                              const char *impl_file, const char *impl_func) {
    memset(ctx, 0, sizeof(*ctx));
    snprintf(ctx->test_id, sizeof(ctx->test_id), "%s", default_id);
    snprintf(ctx->test_name, sizeof(ctx->test_name), "%s", name);
    snprintf(ctx->impl_file, sizeof(ctx->impl_file), "%s", impl_file);
    snprintf(ctx->impl_func, sizeof(ctx->impl_func), "%s", impl_func);
    snprintf(ctx->repo_root, sizeof(ctx->repo_root), "/tmp/avid_layer3_work");
    snprintf(ctx->validation_dir, sizeof(ctx->validation_dir), "/tmp/avid_layer3_work/validation/layer3");
    snprintf(ctx->c_compiler, sizeof(ctx->c_compiler), "cc");

    const char *env_root = getenv("AVID_REPO_ROOT");
    if (env_root && env_root[0]) snprintf(ctx->repo_root, sizeof(ctx->repo_root), "%s", env_root);
    const char *env_val = getenv("AVID_VALIDATION_DIR");
    if (env_val && env_val[0]) snprintf(ctx->validation_dir, sizeof(ctx->validation_dir), "%s", env_val);
    const char *env_cc = getenv("AVID_C_COMPILER");
    if (env_cc && env_cc[0]) snprintf(ctx->c_compiler, sizeof(ctx->c_compiler), "%s", env_cc);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--test-id") == 0 && i + 1 < argc) {
            snprintf(ctx->test_id, sizeof(ctx->test_id), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--repo-root") == 0 && i + 1 < argc) {
            snprintf(ctx->repo_root, sizeof(ctx->repo_root), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--validation-dir") == 0 && i + 1 < argc) {
            snprintf(ctx->validation_dir, sizeof(ctx->validation_dir), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--c-compiler") == 0 && i + 1 < argc) {
            snprintf(ctx->c_compiler, sizeof(ctx->c_compiler), "%s", argv[++i]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ctx->t0);
}

static inline void c_abi_note(c_abi_ctx_t *ctx, const char *msg) {
    if (ctx->note_count < 32) {
        snprintf(ctx->notes[ctx->note_count++], 256, "%s", msg);
    }
}

#define C_ABI_CHECK_TRUE(ctx_ptr, cond, msg) do { \
    (ctx_ptr)->checks_run++; \
    if (!(cond)) { \
        (ctx_ptr)->checks_failed++; \
        char _buf[256]; \
        snprintf(_buf, sizeof(_buf), "[FAIL %s:%d] %s", __FILE__, __LINE__, (msg)); \
        c_abi_note((ctx_ptr), _buf); \
    } \
} while (0)

#define C_ABI_CHECK_EQ_ZU(ctx_ptr, actual, expected, label) do { \
    (ctx_ptr)->checks_run++; \
    size_t _a = (size_t)(actual); \
    size_t _e = (size_t)(expected); \
    if (_a != _e) { \
        (ctx_ptr)->checks_failed++; \
        char _buf[256]; \
        snprintf(_buf, sizeof(_buf), "[FAIL %s:%d] %s actual=%zu expected=%zu", __FILE__, __LINE__, (label), _a, _e); \
        c_abi_note((ctx_ptr), _buf); \
    } \
} while (0)

static inline int c_abi_run_isolated_probe(int (*fn)(void *), void *arg,
                                           int *out_signaled, int *out_sig) {
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
        *out_signaled = 0;
        *out_sig = 0;
        return -1;
    }
    if (pid == 0) {
        /* Silence ASan/stderr crash noise in intentional null-pointer child probe */
        freopen("/dev/null", "w", stderr);
        int rc = fn(arg);
        _exit(rc & 0xFF);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) break;
    }
    if (WIFSIGNALED(status)) {
        *out_signaled = 1;
        *out_sig = WTERMSIG(status);
        return 128 + *out_sig;
    }
    *out_signaled = 0;
    *out_sig = 0;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static inline void c_abi_write_json_escaped(FILE *fp, const char *s) {
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', fp);
            fputc((int)*p, fp);
        } else if (*p == '\n') {
            fputs("\\n", fp);
        } else if (*p == '\r') {
            fputs("\\r", fp);
        } else if (*p == '\t') {
            fputs("\\t", fp);
        } else if (*p >= 0x20) {
            fputc((int)*p, fp);
        }
    }
    fputc('"', fp);
}

static inline int c_abi_finalize(c_abi_ctx_t *ctx, const char *status_str, const char *failure_class) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dur_ms = ((double)(t1.tv_sec - ctx->t0.tv_sec) * 1000.0) +
                    ((double)(t1.tv_nsec - ctx->t0.tv_nsec) / 1e6);

    char raw_dir[600];
    char log_dir[600];
    snprintf(raw_dir, sizeof(raw_dir), "%s/raw_results/set1", ctx->validation_dir);
    snprintf(log_dir, sizeof(log_dir), "%s/logs/set1", ctx->validation_dir);
    c_abi_mkdir_p(raw_dir);
    c_abi_mkdir_p(log_dir);

    char json_path[700];
    char log_path[700];
    snprintf(json_path, sizeof(json_path), "%s/%s.json", raw_dir, ctx->test_id);
    snprintf(log_path, sizeof(log_path), "%s/%s.log", log_dir, ctx->test_id);

    FILE *lf = fopen(log_path, "w");
    if (lf) {
        fprintf(lf, "[%s] %s — status=%s checks_run=%d checks_failed=%d duration_ms=%.3f\n",
                ctx->test_id, ctx->test_name, status_str, ctx->checks_run, ctx->checks_failed, dur_ms);
        fprintf(lf, "Expected: %s\nActual:   %s\n", ctx->expected_summary, ctx->actual_summary);
        for (int i = 0; i < ctx->note_count; ++i) {
            fprintf(lf, "  note[%d]: %s\n", i, ctx->notes[i]);
        }
        fclose(lf);
    }

    FILE *jf = fopen(json_path, "w");
    if (jf) {
        fprintf(jf, "{\n");
        fprintf(jf, "  \"test_id\": "); c_abi_write_json_escaped(jf, ctx->test_id); fprintf(jf, ",\n");
        fprintf(jf, "  \"test_name\": "); c_abi_write_json_escaped(jf, ctx->test_name); fprintf(jf, ",\n");
        fprintf(jf, "  \"test_description\": "); c_abi_write_json_escaped(jf, ctx->description); fprintf(jf, ",\n");
        fprintf(jf, "  \"test_set\": \"set1_c_abi\",\n");
        fprintf(jf, "  \"status\": "); c_abi_write_json_escaped(jf, status_str); fprintf(jf, ",\n");
        fprintf(jf, "  \"source_commit\": \"88df97e86b579b6e2865cc70dc4d0b08225d677b\",\n");
        fprintf(jf, "  \"implementation_file\": "); c_abi_write_json_escaped(jf, ctx->impl_file); fprintf(jf, ",\n");
        fprintf(jf, "  \"implementation_function\": "); c_abi_write_json_escaped(jf, ctx->impl_func); fprintf(jf, ",\n");
        fprintf(jf, "  \"input_contract\": \"Pure C11 (-std=c11) ABI, struct layout, symbol linkage, null safety, buffer ownership\",\n");
        fprintf(jf, "  \"independent_oracle\": \"C11 _Static_assert / offsetof / dlsym vs C++17 layout oracle\",\n");
        fprintf(jf, "  \"fixture\": \"none (direct C11 <-> C++17 binary ABI inspection)\",\n");
        fprintf(jf, "  \"expected\": {\n    \"summary\": "); c_abi_write_json_escaped(jf, ctx->expected_summary); fprintf(jf, "\n  },\n");
        fprintf(jf, "  \"actual\": {\n    \"summary\": "); c_abi_write_json_escaped(jf, ctx->actual_summary);
        fprintf(jf, ",\n    \"checks_run\": \"%d\",\n    \"checks_failed\": \"%d\"\n  },\n", ctx->checks_run, ctx->checks_failed);
        fprintf(jf, "  \"assertion_notes\": [");
        for (int i = 0; i < ctx->note_count; ++i) {
            if (i > 0) fprintf(jf, ", ");
            c_abi_write_json_escaped(jf, ctx->notes[i]);
        }
        fprintf(jf, "],\n");
        fprintf(jf, "  \"evidence\": [\"validation/layer3/logs/set1/%s.log\"],\n", ctx->test_id);
        fprintf(jf, "  \"failure_classification\": "); c_abi_write_json_escaped(jf, failure_class ? failure_class : ""); fprintf(jf, ",\n");
        fprintf(jf, "  \"duration_ms\": %.3f\n}\n", dur_ms);
        fclose(jf);
    }

    printf("[%s] %-28s -> %s (checks=%d, failed=%d, %.2f ms)\n",
           ctx->test_id, ctx->test_name, status_str, ctx->checks_run, ctx->checks_failed, dur_ms);
    return 0;
}

#endif /* AVID_LAYER3_C_ABI_HARNESS_H */
