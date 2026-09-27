import {
  BayesianEngine,
  ROOT_CAUSES,
  OBSERVABLE_SYMPTOMS
} from './bayesian-engine.js';


/**
 * BayesRCA Application Controller
 *
 * Frontend for the final RCAEval-trained Discrete Naive Bayes model.
 *
 * Evidence:
 *   7 telemetry-change features
 *
 * Evidence states:
 *   low | medium | high | unavailable
 *
 * Output:
 *   posterior distribution over
 *   CPU | MEM | DISK | DELAY | LOSS | SOCKET
 */
class AppController {
  constructor() {
    this.engine = new BayesianEngine();

    /*
     * Evidence is stored as:
     *
     * {
     *   cpu_change: "high",
     *   mem_change: "medium",
     *   ...
     * }
     *
     * Missing keys mean unavailable evidence.
     */
    this.evidenceStates = {};

    this.stats = this.loadStats();
    this.history = this.loadHistory();

    this.latestResult = null;

    this.svgTransform = {
      scale: 1,
      translateX: 0,
      translateY: 0
    };

    this.isPanning = false;

    this.panStart = {
      x: 0,
      y: 0
    };

    this.initElements();
    this.renderEvidenceControls();
    this.updateStatsDisplay();
    this.attachEventListeners();
    this.renderInitialDAG();
  }


  // ==========================================================
  // LOCAL STORAGE
  // ==========================================================

  loadStats() {
    try {
      const stored =
        localStorage.getItem('bayesrca_v2_stats');

      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      console.warn(
        'Unable to read BayesRCA statistics.',
        error
      );
    }

    /*
     * Start from zero.
     *
     * We intentionally do NOT ship fake incident counts.
     */
    return {
      totalIncidents: 0,
      highConfidence: 0,
      avgAnalysisTimeMs: 0,
      totalRuns: 0
    };
  }


  saveStats() {
    try {
      localStorage.setItem(
        'bayesrca_v2_stats',
        JSON.stringify(this.stats)
      );
    } catch (error) {
      console.warn(
        'Unable to save BayesRCA statistics.',
        error
      );
    }
  }


  loadHistory() {
    try {
      const stored =
        localStorage.getItem('bayesrca_v2_history');

      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      console.warn(
        'Unable to read BayesRCA history.',
        error
      );
    }

    // No fabricated example incidents.
    return [];
  }


  saveHistory() {
    try {
      localStorage.setItem(
        'bayesrca_v2_history',
        JSON.stringify(this.history)
      );
    } catch (error) {
      console.warn(
        'Unable to save BayesRCA history.',
        error
      );
    }
  }


  // ==========================================================
  // DOM REFERENCES
  // ==========================================================

  initElements() {
    this.inputScreen =
      document.getElementById('input-screen');

    this.resultsScreen =
      document.getElementById('results-screen');

    this.symptomsGrid =
      document.getElementById('symptoms-grid');

    this.selectedCountBadge =
      document.getElementById(
        'selected-count-badge'
      );

    this.analyzeBtn =
      document.getElementById('analyze-btn');

    this.clearAllBtn =
      document.getElementById('clear-all-btn');

    this.newAnalysisBtn =
      document.getElementById(
        'new-analysis-btn'
      );

    this.bottomNewAnalysisBtn =
      document.getElementById(
        'bottom-new-analysis-btn'
      );

    this.computingOverlay =
      document.getElementById(
        'computing-overlay'
      );

    this.computingStep =
      document.getElementById(
        'computing-step'
      );

    this.presetPills =
      document.querySelectorAll(
        '.preset-pill'
      );

    // Existing dashboard stat elements
    this.statTotalIncidents =
      document.getElementById(
        'stat-total-incidents'
      );

    this.statHighSeverity =
      document.getElementById(
        'stat-high-severity'
      );

    this.statAvgTime =
      document.getElementById(
        'stat-avg-time'
      );

    // Results
    this.rankingsTableBody =
      document.getElementById(
        'rankings-table-body'
      );

    this.whyContainer =
      document.getElementById(
        'why-container'
      );

    this.evidenceSummaryContainer =
      document.getElementById(
        'evidence-summary-container'
      );

    this.recommendationsContainer =
      document.getElementById(
        'recommendations-container'
      );

    this.resultsMetaPill =
      document.getElementById(
        'results-meta-pill'
      );

    // SVG
    this.svgContainer =
      document.getElementById(
        'network-viz'
      );

    this.svgCanvas =
      document.getElementById(
        'svg-canvas'
      );

    this.svgViewport =
      document.getElementById(
        'svg-viewport'
      );

    this.zoomInBtn =
      document.getElementById(
        'zoom-in-btn'
      );

    this.zoomOutBtn =
      document.getElementById(
        'zoom-out-btn'
      );

    this.zoomResetBtn =
      document.getElementById(
        'zoom-reset-btn'
      );

    // History
    this.historyDrawer =
      document.getElementById(
        'history-drawer'
      );

    this.historyBackdrop =
      document.getElementById(
        'history-backdrop'
      );

    this.historyBtn =
      document.getElementById(
        'history-btn'
      );

    this.closeHistoryBtn =
      document.getElementById(
        'close-history-btn'
      );

    this.historyList =
      document.getElementById(
        'history-list'
      );

    // Export
    this.exportReportBtn =
      document.getElementById(
        'export-report-btn'
      );
  }


  // ==========================================================
  // DASHBOARD STATS
  // ==========================================================

  updateStatsDisplay() {
    if (this.statTotalIncidents) {
      this.statTotalIncidents.textContent =
        this.stats.totalIncidents;
    }

    /*
     * The old HTML calls this "High Severity".
     * For now the value represents analyses where the highest
     * posterior was >= 60%.
     *
     * index.html will rename the label to High Confidence.
     */
    if (this.statHighSeverity) {
      this.statHighSeverity.textContent =
        this.stats.highConfidence;
    }

    if (this.statAvgTime) {
      this.statAvgTime.textContent =
        `${this.stats.avgAnalysisTimeMs.toFixed(1)} ms`;
    }
  }


  // ==========================================================
  // EVIDENCE INPUT UI
  // ==========================================================

  renderEvidenceControls() {
    if (!this.symptomsGrid) {
      return;
    }

    this.symptomsGrid.innerHTML = '';

    OBSERVABLE_SYMPTOMS.forEach(feature => {
      const card =
        document.createElement('div');

      card.className =
        'symptom-card evidence-state-card';

      card.id =
        `symptom-card-${feature.id}`;

      card.setAttribute(
        'data-id',
        feature.id
      );

      const currentState =
        this.evidenceStates[feature.id]
        ?? 'unavailable';

      card.innerHTML = `
        <div class="symptom-meta evidence-feature-meta">

          <div class="symptom-name">
            <span>${feature.name}</span>
            <span class="category-tag">
              ${feature.category}
            </span>
          </div>

          <div class="symptom-subtext">
            ${feature.description}
          </div>

          <div
            class="evidence-state-selector"
            data-feature="${feature.id}"
          >

            ${this.createStateButton(
              feature.id,
              'low',
              'LOW',
              currentState
            )}

            ${this.createStateButton(
              feature.id,
              'medium',
              'MEDIUM',
              currentState
            )}

            ${this.createStateButton(
              feature.id,
              'high',
              'HIGH',
              currentState
            )}

            ${this.createStateButton(
              feature.id,
              'unavailable',
              'N/A',
              currentState
            )}

          </div>

        </div>
      `;

      const buttons =
        card.querySelectorAll(
          '.evidence-state-btn'
        );

      buttons.forEach(button => {
        button.addEventListener(
          'click',
          event => {
            event.stopPropagation();

            const state =
              button.getAttribute(
                'data-state'
              );

            this.setEvidenceState(
              feature.id,
              state
            );
          }
        );
      });

      this.symptomsGrid.appendChild(
        card
      );
    });

    this.updateSelectedCount();
  }


  createStateButton(
    featureId,
    state,
    label,
    currentState
  ) {
    const active =
      state === currentState
        ? 'active'
        : '';

    return `
      <button
        type="button"
        class="evidence-state-btn state-${state} ${active}"
        data-feature="${featureId}"
        data-state="${state}"
      >
        ${label}
      </button>
    `;
  }


  setEvidenceState(featureId, state) {
    if (state === 'unavailable') {
      delete this.evidenceStates[
        featureId
      ];
    } else {
      this.evidenceStates[
        featureId
      ] = state;
    }

    this.updateEvidenceCard(
      featureId
    );

    this.updateSelectedCount();

    /*
     * Manual edits clear the active preset.
     */
    this.presetPills.forEach(
      pill =>
        pill.classList.remove(
          'active'
        )
    );
  }


  updateEvidenceCard(featureId) {
    const card =
      document.getElementById(
        `symptom-card-${featureId}`
      );

    if (!card) {
      return;
    }

    const currentState =
      this.evidenceStates[featureId]
      ?? 'unavailable';

    card.classList.toggle(
      'selected',
      currentState !== 'unavailable'
    );

    card
      .querySelectorAll(
        '.evidence-state-btn'
      )
      .forEach(button => {
        const buttonState =
          button.getAttribute(
            'data-state'
          );

        button.classList.toggle(
          'active',
          buttonState === currentState
        );
      });
  }


  updateAllEvidenceCards() {
    OBSERVABLE_SYMPTOMS.forEach(
      feature => {
        this.updateEvidenceCard(
          feature.id
        );
      }
    );

    this.updateSelectedCount();
  }


  updateSelectedCount() {
    const count =
      Object.keys(
        this.evidenceStates
      ).length;

    if (this.selectedCountBadge) {
      this.selectedCountBadge.textContent =
        `${count} metric${count === 1 ? '' : 's'} observed`;
    }
  }


  clearEvidence() {
    this.evidenceStates = {};

    this.updateAllEvidenceCards();

    this.presetPills.forEach(
      pill =>
        pill.classList.remove(
          'active'
        )
    );
  }


  // ==========================================================
  // EVENT LISTENERS
  // ==========================================================

  attachEventListeners() {
    if (this.analyzeBtn) {
      this.analyzeBtn.addEventListener(
        'click',
        () => this.executeAnalysis()
      );
    }

    window.addEventListener(
      'keydown',
      event => {
        if (
          (
            event.ctrlKey ||
            event.metaKey
          ) &&
          event.key === 'Enter'
        ) {
          if (
            this.inputScreen &&
            this.inputScreen.style.display
              !== 'none'
          ) {
            this.executeAnalysis();
          }
        }
      }
    );

    if (this.clearAllBtn) {
      this.clearAllBtn.addEventListener(
        'click',
        () => this.clearEvidence()
      );
    }

    this.presetPills.forEach(
      pill => {
        pill.addEventListener(
          'click',
          () => {
            const scenario =
              pill.getAttribute(
                'data-preset'
              );

            this.loadPresetScenario(
              scenario,
              pill
            );
          }
        );
      }
    );

    const goBack = () => {
      if (this.resultsScreen) {
        this.resultsScreen.style.display =
          'none';
      }

      if (this.inputScreen) {
        this.inputScreen.style.display =
          'block';
      }

      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    };

    if (this.newAnalysisBtn) {
      this.newAnalysisBtn.addEventListener(
        'click',
        goBack
      );
    }

    if (this.bottomNewAnalysisBtn) {
      this.bottomNewAnalysisBtn.addEventListener(
        'click',
        goBack
      );
    }

    if (this.zoomInBtn) {
      this.zoomInBtn.addEventListener(
        'click',
        () => this.handleZoom(1.2)
      );
    }

    if (this.zoomOutBtn) {
      this.zoomOutBtn.addEventListener(
        'click',
        () => this.handleZoom(0.8)
      );
    }

    if (this.zoomResetBtn) {
      this.zoomResetBtn.addEventListener(
        'click',
        () => this.resetZoom()
      );
    }

    this.setupSvgPanning();

    if (this.historyBtn) {
      this.historyBtn.addEventListener(
        'click',
        () => this.openHistory()
      );
    }

    if (this.closeHistoryBtn) {
      this.closeHistoryBtn.addEventListener(
        'click',
        () => this.closeHistory()
      );
    }

    if (this.historyBackdrop) {
      this.historyBackdrop.addEventListener(
        'click',
        () => this.closeHistory()
      );
    }

    if (this.exportReportBtn) {
      this.exportReportBtn.addEventListener(
        'click',
        () =>
          this.exportIncidentReport()
      );
    }
  }


  // ==========================================================
  // DEMONSTRATION PRESETS
  // ==========================================================

  loadPresetScenario(
    scenario,
    activePill
  ) {
    this.presetPills.forEach(
      pill =>
        pill.classList.remove(
          'active'
        )
    );

    if (activePill) {
      activePill.classList.add(
        'active'
      );
    }

    /*
     * These presets are demonstration inputs only.
     *
     * They are NOT additional learned model parameters.
     * The actual posterior is still computed entirely from
     * the trained priors and CPTs.
     */
    const presets = {
      'cpu': {
        cpu_change: 'high',
        mem_change: 'medium',
        workload_change: 'high',
        latency50_change: 'medium',
        latency90_change: 'medium'
      },

      'memory': {
        cpu_change: 'high',
        mem_change: 'high',
        workload_change: 'medium',
        latency50_change: 'high',
        latency90_change: 'high'
      },

      'disk': {
        cpu_change: 'medium',
        mem_change: 'medium',
        diskio_change: 'high',
        workload_change: 'medium',
        latency50_change: 'low',
        latency90_change: 'low'
      },

      'delay': {
        cpu_change: 'low',
        mem_change: 'low',
        workload_change: 'low',
        latency50_change: 'high',
        latency90_change: 'high'
      },

      'loss': {
        cpu_change: 'low',
        mem_change: 'medium',
        workload_change: 'low',
        latency50_change: 'medium',
        latency90_change: 'high'
      },

      'socket': {
        cpu_change: 'high',
        mem_change: 'medium',
        diskio_change: 'low',
        socket_change: 'high',
        workload_change: 'high',
        latency50_change: 'medium',
        latency90_change: 'medium'
      },

      'clean': {}
    };

    /*
     * Temporary compatibility with the old HTML preset names.
     *
     * index.html will be updated next.
     */
    const oldPresetAliases = {
      'traffic-spike': 'cpu',
      'db-overload': 'disk',
      'memory-leak': 'memory',
      'dependency-outage': 'delay',
      'network-partition': 'loss'
    };

    const resolvedScenario =
      oldPresetAliases[scenario]
      ?? scenario;

    this.evidenceStates = {
      ...(
        presets[resolvedScenario]
        ?? {}
      )
    };

    this.updateAllEvidenceCards();
  }


  // ==========================================================
  // BAYESIAN ANALYSIS
  // ==========================================================

  async executeAnalysis() {
    const evidenceCount =
      Object.keys(
        this.evidenceStates
      ).length;

    if (evidenceCount === 0) {
      alert(
        'Please provide at least one observed telemetry metric before analyzing.'
      );

      return;
    }

    /*
     * Show overlay first.
     *
     * The delays below are UI presentation delays only and are
     * NOT included in the measured inference runtime.
     */
    if (this.computingOverlay) {
      this.computingOverlay.classList.add(
        'active'
      );
    }

    if (this.computingStep) {
      this.computingStep.textContent =
        'Reading observed telemetry states...';
    }

    await new Promise(
      resolve =>
        setTimeout(resolve, 120)
    );

    if (this.computingStep) {
      this.computingStep.textContent =
        'Applying learned conditional probability tables...';
    }

    await new Promise(
      resolve =>
        setTimeout(resolve, 120)
    );

    if (this.computingStep) {
      this.computingStep.textContent =
        'Computing posterior fault probabilities...';
    }

    /*
     * Measure ONLY actual inference.
     */
    const t0 =
      performance.now();

    let result;

    try {
      result =
        this.engine.runInference({
          ...this.evidenceStates
        });
    } catch (error) {
      console.error(
        'Bayesian inference failed:',
        error
      );

      if (this.computingOverlay) {
        this.computingOverlay.classList.remove(
          'active'
        );
      }

      alert(
        `Inference failed: ${error.message}`
      );

      return;
    }

    const t1 =
      performance.now();

    const computeDurationMs =
      t1 - t0;

    await new Promise(
      resolve =>
        setTimeout(resolve, 100)
    );

    if (this.computingOverlay) {
      this.computingOverlay.classList.remove(
        'active'
      );
    }

    // -------------------------------
    // Real usage statistics
    // -------------------------------

    const previousRuns =
      this.stats.totalRuns;

    this.stats.totalIncidents += 1;
    this.stats.totalRuns += 1;

    /*
     * This is confidence, NOT severity.
     */
    if (
      result.topCause &&
      result.topCause.probability >= 0.60
    ) {
      this.stats.highConfidence += 1;
    }

    this.stats.avgAnalysisTimeMs =
      (
        (
          this.stats.avgAnalysisTimeMs *
          previousRuns
        ) +
        computeDurationMs
      ) /
      this.stats.totalRuns;

    this.saveStats();
    this.updateStatsDisplay();

    // -------------------------------
    // History
    // -------------------------------

    this.history.unshift({
      id:
        `analysis-${Date.now()}`,

      timestamp:
        new Date().toLocaleString(),

      topCause:
        result.topCause?.name
        ?? 'Unknown',

      percentage:
        result.topCause?.percentage
        ?? 0,

      evidence: {
        ...result.evidenceStates
      }
    });

    if (this.history.length > 20) {
      this.history.pop();
    }

    this.saveHistory();

    this.latestResult = result;

    this.displayResults(
      result,
      computeDurationMs
    );
  }


  // ==========================================================
  // RESULTS
  // ==========================================================

  displayResults(
    result,
    computeDurationMs
  ) {
    if (this.inputScreen) {
      this.inputScreen.style.display =
        'none';
    }

    if (this.resultsScreen) {
      this.resultsScreen.style.display =
        'block';
    }

    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });

    if (this.resultsMetaPill) {
      this.resultsMetaPill.textContent =
        `Analyzed ${result.totalEvidenceCount} observed metrics in ${computeDurationMs.toFixed(2)} ms · Discrete Naive Bayes`;
    }

    this.renderDAG(result);

    this.renderRankingsTable(
      result.rankings
    );

    this.renderExplanations(
      result.explanations,
      result.rankings
    );

    this.renderEvidenceSummary(
      result.evidenceStates
    );

    this.renderRecommendations(
      result.topCause
    );
  }


  // ==========================================================
  // POSTERIOR RANKING TABLE
  // ==========================================================

  renderRankingsTable(rankings) {
    if (!this.rankingsTableBody) {
      return;
    }

    this.rankingsTableBody.innerHTML =
      '';

    rankings.forEach(
      (cause, index) => {
        const row =
          document.createElement(
            'tr'
          );

        row.className =
          `rank-${index + 1}`;

        /*
         * Posterior probability is NOT incident severity.
         *
         * Use neutral confidence/rank language.
         */
        let fillClass = 'moderate';

        let badgeLabel =
          index === 0
            ? 'TOP'
            : `#${index + 1}`;

        let badgeStyle =
          'background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25);';

        if (index === 0) {
          fillClass = 'elevated';

          badgeStyle =
            'background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);';
        }

        row.innerHTML = `
          <td style="width:45px;">
            <div class="cause-rank-pill">
              #${index + 1}
            </div>
          </td>

          <td>
            <div class="cause-info-cell">

              <span class="cause-name">
                ${cause.name}

                <span
                  style="
                    font-size:0.65rem;
                    font-weight:700;
                    padding:0.15rem 0.5rem;
                    border-radius:4px;
                    ${badgeStyle}
                  "
                >
                  ${badgeLabel}
                </span>

              </span>

              <span class="cause-desc">
                ${cause.description}
              </span>

            </div>
          </td>

          <td>
            <div class="prior-tag">
              Prior =
              ${(cause.prior * 100).toFixed(1)}%
            </div>
          </td>

          <td>
            <div class="prob-meter-wrapper">

              <div class="prob-track">
                <div
                  class="prob-fill ${fillClass}"
                  style="width:0%;"
                ></div>
              </div>

              <span class="prob-percentage-text">
              ${
              cause.percentage > 0 && cause.percentage < 0.1
              ? '&lt;0.1%'
              : `${cause.percentage.toFixed(1)}%`
              }
              </span>

            </div>
          </td>
        `;

        this.rankingsTableBody.appendChild(
          row
        );

        setTimeout(
          () => {
            const fill =
              row.querySelector(
                '.prob-fill'
              );

            if (fill) {
              fill.style.width =
                `${cause.percentage}%`;
            }
          },
          50
        );
      }
    );
  }


  // ==========================================================
  // EVIDENCE EXPLANATIONS
  // ==========================================================

  renderExplanations(
    explanations,
    rankings
  ) {
    if (!this.whyContainer) {
      return;
    }

    this.whyContainer.innerHTML = '';

    rankings.forEach(
      (cause, index) => {
        const explanation =
          explanations[cause.id];

        if (!explanation) {
          return;
        }

        const card =
          document.createElement(
            'div'
          );

        card.className =
          `why-card ${
            index === 0
              ? 'top-ranked'
              : ''
          }`;

        let evidenceHtml = '';

        if (
          explanation
            .evidenceContributions
            .length > 0
        ) {
          evidenceHtml =
            explanation
              .evidenceContributions
              .map(item => {
                const likelihood =
                  (
                    item.likelihood *
                    100
                  ).toFixed(1);

                return `
                  <span
                    class="why-symptom-tag"
                    title="P(${item.state} | ${cause.name}) = ${likelihood}%"
                  >
                    ${item.name}:
                    ${item.state.toUpperCase()}
                  </span>
                `;
              })
              .join('');
        } else {
          evidenceHtml = `
            <span class="why-symptom-tag empty-notice">
              No observed telemetry evidence.
            </span>
          `;
        }

        card.innerHTML = `
          <div class="why-card-header">

            <div class="why-title">
              <span>
                ${cause.name}
              </span>
            </div>

            <span class="why-percentage-badge">
              ${cause.percentage.toFixed(1)}%
            </span>

          </div>

          <div class="why-contrib-label">
            Observed evidence evaluated for this fault:
          </div>

          <div class="why-symptoms-list">
            ${evidenceHtml}
          </div>
        `;

        this.whyContainer.appendChild(
          card
        );
      }
    );
  }


  // ==========================================================
  // EVIDENCE SUMMARY
  // ==========================================================

  renderEvidenceSummary(
    evidenceStates
  ) {
    if (
      !this.evidenceSummaryContainer
    ) {
      return;
    }

    this.evidenceSummaryContainer.innerHTML =
      '';

    const entries =
      Object.entries(
        evidenceStates
      );

    if (entries.length === 0) {
      this.evidenceSummaryContainer.innerHTML =
        `
          <span
            style="
              color:var(--text-dim);
              font-size:0.85rem;
            "
          >
            No telemetry evidence provided.
          </span>
        `;

      return;
    }

    entries.forEach(
      ([featureId, state]) => {
        const feature =
          OBSERVABLE_SYMPTOMS.find(
            item =>
              item.id === featureId
          );

        const name =
          feature?.name
          ?? featureId;

        const category =
          feature?.category
          ?? 'Telemetry';

        const chip =
          document.createElement(
            'div'
          );

        chip.className =
          'evidence-chip';

        chip.innerHTML = `
          <span>
            ${name}
          </span>

          <span
            style="
              font-size:0.68rem;
              font-weight:800;
              text-transform:uppercase;
              background:rgba(16,185,129,0.14);
              color:#34d399;
              padding:0.1rem 0.4rem;
              border-radius:4px;
            "
          >
            ${state}
          </span>

          <span
            style="
              font-size:0.65rem;
              background:rgba(255,255,255,0.1);
              padding:0.1rem 0.4rem;
              border-radius:4px;
            "
          >
            ${category}
          </span>
        `;

        this.evidenceSummaryContainer
          .appendChild(chip);
      }
    );
  }


  // ==========================================================
  // INVESTIGATION GUIDANCE
  // ==========================================================

  renderRecommendations(topCause) {
    if (
      !this.recommendationsContainer ||
      !topCause
    ) {
      return;
    }

    const stepsHtml =
      topCause
        .investigationSteps
        .map(
          (step, index) => `
            <li class="recom-step-item">

              <div class="step-number">
                ${index + 1}
              </div>

              <div class="step-text">
                ${step}
              </div>

            </li>
          `
        )
        .join('');

    this.recommendationsContainer.innerHTML =
      `
        <div class="recom-header">

          <div class="recom-top-cause-badge">

            <div class="recom-alert-icon">
              <svg
                width="22"
                height="22"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <circle
                  cx="12"
                  cy="12"
                  r="10"
                ></circle>

                <line
                  x1="12"
                  y1="8"
                  x2="12"
                  y2="12"
                ></line>

                <line
                  x1="12"
                  y1="16"
                  x2="12.01"
                  y2="16"
                ></line>
              </svg>
            </div>

            <div>

              <div class="recom-top-title">
                Highest Posterior:
                ${topCause.name}
                (${topCause.percentage.toFixed(1)}%)
              </div>

              <div class="recom-top-subtitle">
                Suggested investigation checklist for the highest-ranked fault hypothesis
              </div>

            </div>

          </div>

          <button
            id="copy-recom-btn"
            class="btn-secondary"
            style="font-size:0.8rem;"
          >
            Copy Checklist
          </button>

        </div>

        <ol class="recom-steps-list">
          ${stepsHtml}
        </ol>

        <div class="recom-disclaimer">

          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <circle
              cx="12"
              cy="12"
              r="10"
            ></circle>

            <line
              x1="12"
              y1="16"
              x2="12"
              y2="12"
            ></line>

            <line
              x1="12"
              y1="8"
              x2="12.01"
              y2="8"
            ></line>
          </svg>

          <span>
            <strong>Investigation advisory:</strong>
            The Bayesian model ranks fault hypotheses from telemetry evidence.
            Confirm the diagnosis using system metrics, logs, or traces before remediation.
          </span>

        </div>
      `;

    const copyButton =
      document.getElementById(
        'copy-recom-btn'
      );

    if (copyButton) {
      copyButton.addEventListener(
        'click',
        () => {
          const text =
            `BayesRCA Investigation Guide\n` +
            `Highest posterior: ${topCause.name} (${topCause.percentage.toFixed(1)}%)\n\n` +
            topCause
              .investigationSteps
              .map(
                (step, index) =>
                  `${index + 1}. ${step}`
              )
              .join('\n');

          navigator.clipboard
            .writeText(text)
            .then(() => {
              copyButton.textContent =
                'Copied';

              setTimeout(
                () => {
                  copyButton.textContent =
                    'Copy Checklist';
                },
                1500
              );
            });
        }
      );
    }
  }


  // ==========================================================
  // BAYESIAN NETWORK VISUALIZATION
  // ==========================================================

  renderInitialDAG() {
    this.renderDAG(null);
  }


  renderDAG(result) {
    if (!this.svgViewport) {
      return;
    }

    /*
     * Correct model structure:
     *
     *              FAULT TYPE
     *                  |
     *       -------------------------
     *       |    |    |    |   ...
     *      CPU  MEM  DISK ...
     *
     * Fault Type is ONE categorical latent variable.
     *
     * CPU/MEM/DISK/DELAY/LOSS/SOCKET are its possible states,
     * not six independent Boolean root nodes.
     */

    const svgWidth = 1420;
    const svgHeight = 470;

    const faultNode = {
      x: 710,
      y: 105
    };

    const featureY = 350;

    const featureSpacing = 180;

    const startX =
      (
        svgWidth -
        featureSpacing *
        (
          OBSERVABLE_SYMPTOMS.length -
          1
        )
      ) / 2;

    const featureCoords = {};

    OBSERVABLE_SYMPTOMS.forEach(
      (feature, index) => {
        featureCoords[feature.id] = {
          x:
            startX +
            index * featureSpacing,

          y: featureY
        };
      }
    );

    const topCause =
      result?.topCause
      ?? null;

    const evidenceStates =
      result?.evidenceStates
      ?? {};

    let svgContent = `
      <defs>

        <marker
          id="arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path
            d="M 0 1.5 L 8 5 L 0 8.5 z"
            fill="rgba(255,255,255,0.22)"
          />
        </marker>

        <marker
          id="arrow-active"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path
            d="M 0 1.5 L 8 5 L 0 8.5 z"
            fill="#38bdf8"
          />
        </marker>

        <filter
          id="glow-primary"
          x="-30%"
          y="-30%"
          width="160%"
          height="160%"
        >
          <feGaussianBlur
            stdDeviation="5"
            result="blur"
          />

          <feComposite
            in="SourceGraphic"
            in2="blur"
            operator="over"
          />
        </filter>

        <filter
          id="glow-evidence"
          x="-30%"
          y="-30%"
          width="160%"
          height="160%"
        >
          <feGaussianBlur
            stdDeviation="4"
            result="blur"
          />

          <feComposite
            in="SourceGraphic"
            in2="blur"
            operator="over"
          />
        </filter>

      </defs>


      <!-- MODEL LAYER -->

      <rect
        x="20"
        y="25"
        width="${svgWidth - 40}"
        height="170"
        rx="14"
        fill="rgba(56,189,248,0.025)"
        stroke="rgba(56,189,248,0.12)"
        stroke-dasharray="4 4"
      />

      <text
        x="38"
        y="50"
        fill="#38bdf8"
        font-size="11"
        font-weight="700"
        letter-spacing="1"
      >
        LATENT CATEGORICAL VARIABLE
      </text>


      <!-- EVIDENCE LAYER -->

      <rect
        x="20"
        y="260"
        width="${svgWidth - 40}"
        height="175"
        rx="14"
        fill="rgba(16,185,129,0.02)"
        stroke="rgba(16,185,129,0.10)"
        stroke-dasharray="4 4"
      />

      <text
        x="38"
        y="285"
        fill="#34d399"
        font-size="11"
        font-weight="700"
        letter-spacing="1"
      >
        OBSERVED TELEMETRY FEATURES
      </text>


      <!-- EDGES -->

      <g class="dag-edges">
    `;

    OBSERVABLE_SYMPTOMS.forEach(
      feature => {
        const target =
          featureCoords[feature.id];

        const observed =
          evidenceStates[feature.id]
          !== undefined;

        const pathData =
          `M ${faultNode.x} ${faultNode.y + 38} ` +
          `C ${faultNode.x} 220, ` +
          `${target.x} 235, ` +
          `${target.x} ${target.y - 28}`;

        svgContent += `
          <path
            d="${pathData}"
            class="dag-edge ${observed ? 'active' : ''}"
            marker-end="${
              observed
                ? 'url(#arrow-active)'
                : 'url(#arrow)'
            }"
          />
        `;
      }
    );

    svgContent += `
      </g>


      <!-- FAULT TYPE NODE -->

      <g
        class="dag-node"
        id="node-fault-type"
      >

        <rect
          x="${faultNode.x - 125}"
          y="${faultNode.y - 38}"
          width="250"
          height="76"
          rx="13"
          fill="#0f172a"
          stroke="${
            result
              ? '#38bdf8'
              : 'rgba(56,189,248,0.35)'
          }"
          stroke-width="${
            result
              ? '2'
              : '1'
          }"
          ${
            result
              ? 'filter="url(#glow-primary)"'
              : ''
          }
        />

        <text
          x="${faultNode.x}"
          y="${faultNode.y - 9}"
          fill="#94a3b8"
          font-size="10"
          font-weight="700"
          text-anchor="middle"
          letter-spacing="1"
        >
          FAULT TYPE
        </text>

        <text
          x="${faultNode.x}"
          y="${faultNode.y + 12}"
          fill="#f8fafc"
          font-size="15"
          font-weight="800"
          text-anchor="middle"
        >
          ${
            topCause
              ? topCause.name
              : 'CPU · MEM · DISK · DELAY · LOSS · SOCKET'
          }
        </text>
    `;

    if (topCause) {
      svgContent += `
        <text
          x="${faultNode.x}"
          y="${faultNode.y + 29}"
          fill="#38bdf8"
          font-size="11"
          font-weight="700"
          text-anchor="middle"
        >
          Highest posterior:
          ${topCause.percentage.toFixed(1)}%
        </text>
      `;
    }

    svgContent += `
      </g>


      <!-- FEATURE NODES -->

      <g class="dag-nodes">
    `;

    OBSERVABLE_SYMPTOMS.forEach(
      feature => {
        const coord =
          featureCoords[feature.id];

        const state =
          evidenceStates[feature.id];

        const observed =
          state !== undefined;

        const width = 150;
        const height = 58;

        const x =
          coord.x -
          width / 2;

        const y =
          coord.y -
          height / 2;

        svgContent += `
          <g
            class="dag-node"
            id="node-${feature.id}"
          >

            <rect
              x="${x}"
              y="${y}"
              width="${width}"
              height="${height}"
              rx="10"
              fill="${
                observed
                  ? 'rgba(16,185,129,0.12)'
                  : '#0a0f1d'
              }"
              stroke="${
                observed
                  ? '#10b981'
                  : 'rgba(255,255,255,0.10)'
              }"
              stroke-width="${
                observed
                  ? '2'
                  : '1'
              }"
              ${
                observed
                  ? 'filter="url(#glow-evidence)"'
                  : ''
              }
            />

            <text
              x="${coord.x}"
              y="${coord.y - 5}"
              fill="${
                observed
                  ? '#f8fafc'
                  : '#94a3b8'
              }"
              font-size="10.5"
              font-weight="700"
              text-anchor="middle"
            >
              ${feature.shortName}
            </text>

            <text
              x="${coord.x}"
              y="${coord.y + 14}"
              fill="${
                observed
                  ? '#34d399'
                  : '#64748b'
              }"
              font-size="9.5"
              font-weight="800"
              text-anchor="middle"
            >
              ${
                observed
                  ? state.toUpperCase()
                  : 'UNAVAILABLE'
              }
            </text>

          </g>
        `;
      }
    );

    svgContent += `
      </g>
    `;

    this.svgViewport.innerHTML =
      svgContent;
  }


  // ==========================================================
  // PAN / ZOOM
  // ==========================================================

  setupSvgPanning() {
    if (!this.svgCanvas) {
      return;
    }

    this.svgCanvas.addEventListener(
      'mousedown',
      event => {
        if (
          event.target.closest(
            '.dag-node'
          )
        ) {
          return;
        }

        this.isPanning = true;

        this.panStart = {
          x:
            event.clientX -
            this.svgTransform.translateX,

          y:
            event.clientY -
            this.svgTransform.translateY
        };

        this.svgCanvas.style.cursor =
          'grabbing';
      }
    );

    window.addEventListener(
      'mousemove',
      event => {
        if (!this.isPanning) {
          return;
        }

        this.svgTransform.translateX =
          event.clientX -
          this.panStart.x;

        this.svgTransform.translateY =
          event.clientY -
          this.panStart.y;

        this.updateSvgTransform();
      }
    );

    window.addEventListener(
      'mouseup',
      () => {
        if (!this.isPanning) {
          return;
        }

        this.isPanning = false;

        this.svgCanvas.style.cursor =
          'default';
      }
    );

    this.svgCanvas.addEventListener(
      'wheel',
      event => {
        event.preventDefault();

        const zoomFactor =
          event.deltaY < 0
            ? 1.1
            : 0.9;

        this.handleZoom(
          zoomFactor
        );
      },
      {
        passive: false
      }
    );
  }


  handleZoom(factor) {
    this.svgTransform.scale =
      Math.max(
        0.6,
        Math.min(
          2.5,
          this.svgTransform.scale *
          factor
        )
      );

    this.updateSvgTransform();
  }


  resetZoom() {
    this.svgTransform = {
      scale: 1,
      translateX: 0,
      translateY: 0
    };

    this.updateSvgTransform();
  }


  updateSvgTransform() {
    if (!this.svgViewport) {
      return;
    }

    this.svgViewport.setAttribute(
      'transform',
      `translate(${this.svgTransform.translateX}, ${this.svgTransform.translateY}) scale(${this.svgTransform.scale})`
    );
  }


  // ==========================================================
  // HISTORY
  // ==========================================================

  openHistory() {
    this.renderHistoryList();

    if (this.historyDrawer) {
      this.historyDrawer.classList.add(
        'active'
      );
    }

    if (this.historyBackdrop) {
      this.historyBackdrop.classList.add(
        'active'
      );
    }
  }


  closeHistory() {
    if (this.historyDrawer) {
      this.historyDrawer.classList.remove(
        'active'
      );
    }

    if (this.historyBackdrop) {
      this.historyBackdrop.classList.remove(
        'active'
      );
    }
  }


  renderHistoryList() {
    if (!this.historyList) {
      return;
    }

    this.historyList.innerHTML = '';

    if (this.history.length === 0) {
      this.historyList.innerHTML = `
        <div
          style="
            color:var(--text-dim);
            font-size:0.85rem;
            padding:1rem;
          "
        >
          No prior analyses.
        </div>
      `;

      return;
    }

    this.history.forEach(
      item => {
        const element =
          document.createElement(
            'div'
          );

        element.className =
          'history-item';

        const evidenceCount =
          Object.keys(
            item.evidence ?? {}
          ).length;

        element.innerHTML = `
          <div class="history-item-top">

            <span class="history-cause">
              ${item.topCause}
            </span>

            <span class="history-prob">
              ${Number(item.percentage).toFixed(1)}%
            </span>

          </div>

          <div class="history-time">
            ${item.timestamp}
            ·
            ${evidenceCount}
            observed metric${
              evidenceCount === 1
                ? ''
                : 's'
            }
          </div>
        `;

        element.addEventListener(
          'click',
          () => {
            this.evidenceStates = {
              ...(item.evidence ?? {})
            };

            this.updateAllEvidenceCards();

            this.closeHistory();

            if (evidenceCount > 0) {
              this.executeAnalysis();
            }
          }
        );

        this.historyList.appendChild(
          element
        );
      }
    );
  }


  // ==========================================================
  // REPORT EXPORT
  // ==========================================================

  exportIncidentReport() {
    if (!this.latestResult) {
      return;
    }

    const result =
      this.latestResult;

    const metadata =
      this.engine.getModelMetadata();

    const evidenceLines =
      Object.entries(
        result.evidenceStates
      )
        .map(
          ([featureId, state]) => {
            const feature =
              OBSERVABLE_SYMPTOMS.find(
                item =>
                  item.id === featureId
              );

            return (
              `- ${feature?.name ?? featureId}: ` +
              `${state.toUpperCase()}`
            );
          }
        )
        .join('\n');

    const rankingLines =
      result.rankings
        .map(
          (cause, index) =>
            `${index + 1}. ${cause.name}: ${cause.percentage.toFixed(1)}%`
        )
        .join('\n');

    const report =
`# BayesRCA Incident Diagnosis Report

Generated: ${new Date().toISOString()}

## Model

Model: Discrete Naive Bayes
Dataset: ${metadata.dataset ?? 'RCAEval RE1 + RE2'}

## Highest Posterior Fault

Fault: ${result.topCause.name}
Posterior Probability: ${result.topCause.percentage.toFixed(1)}%
Prior Probability: ${(result.topCause.prior * 100).toFixed(1)}%

${result.topCause.description}

## Posterior Ranking

${rankingLines}

## Observed Telemetry Evidence

${evidenceLines}

## Suggested Investigation Checklist

${result.topCause.investigationSteps
  .map(
    (step, index) =>
      `${index + 1}. ${step}`
  )
  .join('\n')}

## Interpretation

The probabilities above are Bayesian posterior probabilities under the trained Discrete Naive Bayes model. They rank the six RCAEval fault hypotheses given the observed telemetry states. Missing telemetry is omitted from the likelihood calculation.

The investigation checklist is operational guidance and is not itself learned by the Bayesian model.

---
Generated by BayesRCA
`;

    const blob =
      new Blob(
        [report],
        {
          type:
            'text/markdown'
        }
      );

    const url =
      URL.createObjectURL(
        blob
      );

    const anchor =
      document.createElement(
        'a'
      );

    anchor.href = url;

    anchor.download =
      `BayesRCA-Analysis-${Date.now()}.md`;

    document.body.appendChild(
      anchor
    );

    anchor.click();

    document.body.removeChild(
      anchor
    );

    URL.revokeObjectURL(
      url
    );
  }
}


// ============================================================
// BOOTSTRAP
// ============================================================

document.addEventListener(
  'DOMContentLoaded',
  () => {
    window.app =
      new AppController();
  }
);