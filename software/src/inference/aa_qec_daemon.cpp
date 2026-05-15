/* ===========================================================================
 * avid_daemon.cpp — AVID Slow-Path Main Loop
 * Project : AVID (Adaptive Variational Inference Decoder)
 *
 * This process runs on the ARM Cortex-A9/A53 under Linux.
 * It is the "brain" of the adaptive system:
 *
 *   while (true) {
 *     1. Read new syndrome rounds from DMA receive buffer (non-blocking)
 *     2. Push syndromes into ring buffer (last W=10,000 rounds)
 *     3. Run L-BFGS optimiser for up to 50 iterations (~5ms budget)
 *     4. Inject updated per-qubit LLRs → FPGA via AXI-Lite
 *     5. Log noise parameters and epoch count for offline analysis
 *   }
 *
 * COMMANDMENT #9 (No OS Jitter in Data Path):
 *   This daemon NEVER gates the FPGA decoder.  The FPGA runs continuously.
 *   The 10ms inference loop is completely asynchronous from the 68ns decode.
 *
 * COMMANDMENT #3 (No dynamic allocation in hot loop):
 *   ring_buffer_t, NoiseModel, LBFGSOptimizer, WeightInjector are all
 *   allocated ONCE as locals or globals before the main loop begins.
 * =========================================================================*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <signal.h>
#include <time.h>

#include "ring_buffer.h"
#include "noise_model.h"
#include "lbfgs_optimizer.h"
#include "weight_injector.h"
#include "../driver/weight_reg_driver.h"
#include "../driver/dma_manager.h"

/* -------------------------------------------------------------------------
 * Global pre-allocated objects (no heap allocation in main loop)
 * -------------------------------------------------------------------------*/
static ring_buffer_t   g_ring;
static NoiseModel      g_model;
static LBFGSOptimizer  g_optim;
static wrd_handle_t    g_wrd;
static dma_handle_t    g_dma;

/* Receive buffer: 4 bytes per round (correction word), 4096 rounds max */
static uint32_t g_correction_buf[4096];
/* Syndrome accumulation buffer from DMA */
static uint16_t g_syndrome_chunk[4096];

static volatile int g_running = 1;

static void sigint_handler(int) { g_running = 0; }

/* -------------------------------------------------------------------------
 * Logging (writes to stdout — redirect to file in production)
 * -------------------------------------------------------------------------*/
static void log_noise_state(int epoch, double nll) {
    /* Print single-qubit rates and top-3 crosstalk parameters */
    printf("[epoch %6d] NLL=%.4f  p=[", epoch, nll);
    for (int i = 0; i < NM_N_QUBIT; i++)
        printf("%.4f%s", g_model.effective_p(i), i < NM_N_QUBIT-1 ? "," : "");
    printf("]\n");
    fflush(stdout);
}

/* -------------------------------------------------------------------------
 * main()
 * -------------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
    signal(SIGINT,  sigint_handler);
    signal(SIGTERM, sigint_handler);

    printf("AVID Adaptive Daemon — ARM Slow-Path\n");
    printf("  Ring window  : W = %d syndrome rounds\n", RING_WINDOW);
    printf("  L-BFGS iters : max %d per update cycle\n", LBFGS_MAX_ITER);
    printf("  Parameters   : %d (single) + %d (crosstalk) = %d\n",
           NM_N_QUBIT, NM_N_CROSSTALK, NM_N_PARAMS);

    /* Open drivers */
    if (wrd_open(&g_wrd, WRD_BASE_ADDR) < 0) {
        fprintf(stderr, "[daemon] Cannot open weight register driver\n");
        return 1;
    }
    if (dma_open(&g_dma, DMA_BASE_ADDR) < 0) {
        fprintf(stderr, "[daemon] Cannot open DMA driver\n");
        wrd_close(&g_wrd);
        return 1;
    }

    /* Initialise noise model at p0 = 0.01 and inject default weights */
    ring_init(&g_ring);
    g_model.init(0.01);
    WeightInjector injector(&g_wrd);
    injector.inject_default(0.01f);
    printf("[daemon] Default weights injected. Beginning adaptive loop.\n");

    /* -----------------------------------------------------------------------
     * Main adaptive loop
     * ----------------------------------------------------------------------- */
    int    update_epoch      = 0;
    int    syndrome_batch_sz = 100;  /* Rounds per DMA fetch */
    double nll               = 0.0;

    while (g_running) {

        /* ------------------------------------------------------------------
         * STEP 1: Fetch new syndromes (non-blocking — return immediately
         * if DMA not ready).  The FPGA decoder never waits for us.
         * ------------------------------------------------------------------ */
        int n_new = 0;
        /* In simulation context: read from a pre-loaded syndrome file.
         * On hardware: poll DMA completion register. */
        if (dma_wait_complete(&g_dma, /*timeout_us=*/100) == 0) {
            dma_read_corrections(&g_dma, g_correction_buf,
                                 syndrome_batch_sz * sizeof(uint32_t));
            n_new = syndrome_batch_sz;
        }

        /* ------------------------------------------------------------------
         * STEP 2: Push syndromes into ring buffer
         * ------------------------------------------------------------------ */
        for (int i = 0; i < n_new; i++) {
            syndrome_round_t sr;
            /* Syndrome bits are packed in the correction word's upper half
             * (the lower half carries the Pauli correction).
             * On hardware, a separate DMA channel carries syndrome echoes.
             * In simulation: decode from syndrome binary file. */
            uint16_t word = (uint16_t)(g_syndrome_chunk[i]);
            for (int c = 0; c < RING_N_CHECK_X; c++)
                sr.x[c] = (word >> c) & 1;
            for (int c = 0; c < RING_N_CHECK_Z; c++)
                sr.z[c] = (word >> (RING_N_CHECK_X + c)) & 1;
            ring_push(&g_ring, &sr);
        }

        /* ------------------------------------------------------------------
         * STEP 3: Inference — only run when ring buffer has enough history
         * ------------------------------------------------------------------ */
        if (g_ring.count >= 500) {
            nll = g_optim.step(g_model, g_ring);

            /* ------------------------------------------------------------------
             * STEP 4: Inject updated LLRs into FPGA shadow bank and commit
             * ------------------------------------------------------------------ */
            injector.inject(g_model);
            update_epoch++;

            if (update_epoch % 100 == 0)
                log_noise_state(update_epoch, nll);
        }

        /* Yield: allow kernel scheduler to run between update cycles.
         * This prevents the daemon from monopolising CPU time while the
         * FPGA runs deterministically at 250 MHz. */
        usleep(1000);  /* 1 ms sleep — acceptable for 10ms update target */
    }

    /* Cleanup */
    printf("[daemon] Shutting down after %d weight updates.\n", update_epoch);
    wrd_close(&g_wrd);
    dma_close(&g_dma);
    return 0;
}
