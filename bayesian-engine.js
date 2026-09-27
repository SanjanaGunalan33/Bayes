/**
 * BayesRCA Bayesian Inference Engine
 *
 * Implements the same Discrete Naive Bayes inference used by the
 * final Python RCAEval model.
 *
 * Model:
 *   Fault Type
 *      ├── CPU change
 *      ├── Memory change
 *      ├── Disk I/O change
 *      ├── Socket change
 *      ├── Workload change
 *      ├── P50 latency change
 *      └── P90 latency change
 *
 * Evidence states:
 *   low | medium | high | unavailable
 *
 * Missing/unavailable evidence is ignored during inference,
 * exactly as in the Python model.
 */

import { TRAINED_MODEL_PARAMS } from './trained_model_params.js';


/* ============================================================
   ROOT-CAUSE DEFINITIONS
   ============================================================ */

export const ROOT_CAUSES = [
  {
    id: 'cpu',
    name: 'CPU Fault',
    code: 'CPU',
    prior: TRAINED_MODEL_PARAMS.priors.cpu,
    description:
      'Telemetry pattern associated with CPU-related fault injection.',
    investigationSteps: [
      'Inspect CPU utilization for the most affected service.',
      'Compare service CPU usage before and after the incident.',
      'Check for CPU saturation, throttling, or abnormal processing load.',
      'Identify the service showing the strongest CPU change.',
      'Correlate the CPU change with latency and workload telemetry.'
    ]
  },

  {
    id: 'mem',
    name: 'Memory Fault',
    code: 'MEM',
    prior: TRAINED_MODEL_PARAMS.priors.mem,
    description:
      'Telemetry pattern associated with memory-related fault injection.',
    investigationSteps: [
      'Inspect memory utilization for the most affected service.',
      'Compare memory behaviour before and after the incident.',
      'Check for abnormal memory growth or memory pressure.',
      'Inspect container or process memory limits where applicable.',
      'Correlate memory changes with CPU and latency telemetry.'
    ]
  },

  {
    id: 'disk',
    name: 'Disk I/O Fault',
    code: 'DISK',
    prior: TRAINED_MODEL_PARAMS.priors.disk,
    description:
      'Telemetry pattern associated with disk I/O fault injection.',
    investigationSteps: [
      'Inspect disk I/O telemetry for the affected services.',
      'Identify the service with the strongest disk I/O change.',
      'Check storage latency, throughput, and I/O contention.',
      'Compare disk behaviour before and after the incident.',
      'Correlate disk changes with service latency.'
    ]
  },

  {
    id: 'delay',
    name: 'Delay Fault',
    code: 'DELAY',
    prior: TRAINED_MODEL_PARAMS.priors.delay,
    description:
      'Telemetry pattern associated with injected service or network delay.',
    investigationSteps: [
      'Inspect P50 and P90 latency changes across services.',
      'Identify the service with the strongest latency increase.',
      'Compare median latency with tail latency behaviour.',
      'Inspect upstream and downstream service communication.',
      'Check whether latency increased without corresponding resource saturation.'
    ]
  },

  {
    id: 'loss',
    name: 'Packet Loss Fault',
    code: 'LOSS',
    prior: TRAINED_MODEL_PARAMS.priors.loss,
    description:
      'Telemetry pattern associated with packet-loss fault injection.',
    investigationSteps: [
      'Inspect network and service latency telemetry.',
      'Pay particular attention to P90 tail-latency changes.',
      'Check packet retransmissions and communication failures where available.',
      'Identify services showing abnormal network behaviour.',
      'Compare the incident pattern with ordinary delay behaviour.'
    ]
  },

  {
    id: 'socket',
    name: 'Socket Fault',
    code: 'SOCKET',
    prior: TRAINED_MODEL_PARAMS.priors.socket,
    description:
      'Telemetry pattern associated with socket-related fault injection.',
    investigationSteps: [
      'Inspect socket telemetry for the most affected service.',
      'Identify abnormal changes in active or failed socket behaviour.',
      'Check service connection establishment and connection handling.',
      'Correlate socket changes with latency and CPU behaviour.',
      'Inspect the affected service for connection-level failures.'
    ]
  }
];


/* ============================================================
   OBSERVABLE EVIDENCE
   ============================================================ */

/**
 * These are the seven features used by the final trained model.
 *
 * They are not manually invented symptoms. They correspond directly
 * to the telemetry features extracted by train_bayes_rca.py.
 */
export const OBSERVABLE_SYMPTOMS = [
  {
    id: 'cpu_change',
    name: 'CPU Change',
    shortName: 'CPU',
    category: 'Resource',
    icon: 'cpu',
    description: 'Strongest service-level CPU change after fault injection.'
  },

  {
    id: 'mem_change',
    name: 'Memory Change',
    shortName: 'Memory',
    category: 'Resource',
    icon: 'database',
    description: 'Strongest service-level memory change after fault injection.'
  },

  {
    id: 'diskio_change',
    name: 'Disk I/O Change',
    shortName: 'Disk I/O',
    category: 'Resource',
    icon: 'hard-drive',
    description: 'Strongest service-level disk I/O change after fault injection.'
  },

  {
    id: 'socket_change',
    name: 'Socket Change',
    shortName: 'Socket',
    category: 'Network',
    icon: 'wifi',
    description: 'Strongest service-level socket telemetry change after fault injection.'
  },

  {
    id: 'workload_change',
    name: 'Workload Change',
    shortName: 'Workload',
    category: 'Traffic',
    icon: 'activity',
    description: 'Strongest service-level workload change after fault injection.'
  },

  {
    id: 'latency50_change',
    name: 'P50 Latency Change',
    shortName: 'P50 Latency',
    category: 'Performance',
    icon: 'clock',
    description: 'Strongest service-level median latency change after fault injection.'
  },

  {
    id: 'latency90_change',
    name: 'P90 Latency Change',
    shortName: 'P90 Latency',
    category: 'Performance',
    icon: 'clock',
    description: 'Strongest service-level tail-latency change after fault injection.'
  }
];


/*
 * Kept as an empty export temporarily so the existing app.js import
 * does not fail while the frontend is being migrated away from the
 * old three-layer DAG.
 */
export const INTERMEDIATE_EFFECTS = [];


/* ============================================================
   HELPER FUNCTIONS
   ============================================================ */

/**
 * Converts a continuous telemetry-change value into the exact
 * low / medium / high state used during model training.
 */
export function discretizeFeature(featureId, value) {
  if (
    value === null ||
    value === undefined ||
    value === '' ||
    Number.isNaN(Number(value))
  ) {
    return null;
  }

  const thresholds = TRAINED_MODEL_PARAMS.thresholds[featureId];

  if (!thresholds) {
    throw new Error(`Unknown feature: ${featureId}`);
  }

  const numericValue = Number(value);

  if (numericValue <= thresholds.low_medium) {
    return 'low';
  }

  if (numericValue <= thresholds.medium_high) {
    return 'medium';
  }

  return 'high';
}


/**
 * Stable log-sum-exp normalization.
 *
 * Converts log posterior scores into probabilities without
 * numerical underflow.
 */
function normalizeLogScores(logScores) {
  const values = Object.values(logScores);

  if (values.length === 0) {
    return {};
  }

  const maxLogScore = Math.max(...values);

  const expScores = {};
  let denominator = 0;

  for (const [classId, score] of Object.entries(logScores)) {
    const expScore = Math.exp(score - maxLogScore);
    expScores[classId] = expScore;
    denominator += expScore;
  }

  const probabilities = {};

  for (const [classId, expScore] of Object.entries(expScores)) {
    probabilities[classId] =
      denominator > 0
        ? expScore / denominator
        : 0;
  }

  return probabilities;
}


/* ============================================================
   BAYESIAN ENGINE
   ============================================================ */

export class BayesianEngine {
  constructor() {
    this.model = TRAINED_MODEL_PARAMS;

    this.rootCauses = ROOT_CAUSES;
    this.symptoms = OBSERVABLE_SYMPTOMS;

    // Compatibility only while app.js is being migrated.
    this.intermediateEffects = INTERMEDIATE_EFFECTS;
  }


  /**
   * Run Bayesian inference.
   *
   * Preferred evidence format:
   *
   * {
   *   cpu_change: "high",
   *   mem_change: "medium",
   *   diskio_change: null,
   *   socket_change: "low",
   *   workload_change: "medium",
   *   latency50_change: "high",
   *   latency90_change: "high"
   * }
   *
   * null / undefined / "unavailable" means the metric is missing
   * and therefore contributes NO likelihood term.
   */
  runInference(evidenceMap = {}) {
    const observedEvidence = this.prepareEvidence(evidenceMap);

    const logScores = {};

    /*
     * Naive Bayes:
     *
     * log P(C | E) ∝
     * log P(C) +
     * Σ log P(E_i | C)
     *
     * Only observed features are included.
     */
    for (const classId of this.model.classes) {
      const prior = this.model.priors[classId];

      if (!(prior > 0)) {
        throw new Error(
          `Invalid prior probability for class: ${classId}`
        );
      }

      let score = Math.log(prior);

      for (const [featureId, state] of Object.entries(observedEvidence)) {
        const featureCPT = this.model.cpts[featureId];

        if (!featureCPT) {
          continue;
        }

        const classCPT = featureCPT[classId];

        if (!classCPT) {
          continue;
        }

        const likelihood = classCPT[state];

        if (!(likelihood > 0)) {
          continue;
        }

        score += Math.log(likelihood);
      }

      logScores[classId] = score;
    }

    const probabilities = normalizeLogScores(logScores);

    const rankings = this.rootCauses
      .map(rootCause => {
        const posterior =
          probabilities[rootCause.id] ?? 0;

        return {
          id: rootCause.id,
          name: rootCause.name,
          code: rootCause.code,
          prior: rootCause.prior,

          probability: posterior,

          percentage:
            posterior * 100,

          description: rootCause.description,
          investigationSteps: rootCause.investigationSteps
        };
      })
      .sort(
        (a, b) =>
          b.probability - a.probability
      );

    const explanations =
      this.computeExplanations(
        observedEvidence,
        rankings
      );

    const activePaths =
      this.computeActiveGraphTopology(
        observedEvidence,
        rankings
      );

    return {
      activeEvidence: Object.keys(observedEvidence),

      evidenceStates: {
        ...observedEvidence
      },

      rankings,

      topCause:
        rankings.length > 0
          ? rankings[0]
          : null,

      explanations,

      activePaths,

      totalEvidenceCount:
        Object.keys(observedEvidence).length
    };
  }


  /**
   * Validates and normalizes evidence.
   *
   * Accepted values:
   *   low
   *   medium
   *   high
   *   unavailable
   *   null
   *
   * Numeric values are also accepted and automatically discretized
   * using the train-derived thresholds.
   */
  prepareEvidence(evidenceMap) {
    const observed = {};

    for (const feature of this.model.features) {
      let value = evidenceMap[feature];

      if (
        value === undefined ||
        value === null ||
        value === '' ||
        value === 'unavailable'
      ) {
        continue;
      }

      if (typeof value === 'string') {
        value = value.toLowerCase().trim();
      }

      if (this.model.states.includes(value)) {
        observed[feature] = value;
        continue;
      }

      const numericValue = Number(value);

      if (!Number.isNaN(numericValue)) {
        const state =
          discretizeFeature(
            feature,
            numericValue
          );

        if (state !== null) {
          observed[feature] = state;
        }

        continue;
      }

      throw new Error(
        `Invalid evidence value for ${feature}: ${value}`
      );
    }

    return observed;
  }


  /**
   * Shows how each observed evidence item contributes to each
   * class score.
   *
   * This is evidence contribution, not a claim of causal proof.
   */
  computeExplanations(observedEvidence, rankings) {
    const explanations = {};

    for (const rootCause of rankings) {
      const contributions = [];

      for (
        const [featureId, state]
        of Object.entries(observedEvidence)
      ) {
        const feature =
          this.symptoms.find(
            item => item.id === featureId
          );

        const likelihood =
          this.model.cpts?.[featureId]
            ?.[rootCause.id]
            ?.[state];

        if (likelihood === undefined) {
          continue;
        }

        contributions.push({
          id: featureId,

          name:
            feature?.name ??
            featureId,

          category:
            feature?.category ??
            'Telemetry',

          state,

          likelihood,

          logLikelihood:
            Math.log(likelihood)
        });
      }

      /*
       * Larger class-conditional likelihood = stronger compatibility
       * between that evidence state and the candidate fault.
       */
      contributions.sort(
        (a, b) =>
          b.likelihood - a.likelihood
      );

      explanations[rootCause.id] = {
        rootCauseId: rootCause.id,
        rootCauseName: rootCause.name,

        percentage:
          rootCause.percentage,

        contributingSymptoms:
          contributions,

        evidenceContributions:
          contributions,

        hasStrongEvidence:
          contributions.length > 0
      };
    }

    return explanations;
  }


  /**
   * Returns topology information for the new visualization.
   *
   * The trained model is a Naive Bayes network:
   *
   *               Fault Type
   *              /    |     \
   *           evidence features
   *
   * There is no learned intermediate-effect layer.
   */
  computeActiveGraphTopology(observedEvidence, rankings) {
    const activeNodes = new Set();
    const activeEdges = new Set();

    /*
     * The highest-posterior class is highlighted as the inferred
     * state of the categorical Fault Type variable.
     */
    if (rankings.length > 0) {
      activeNodes.add(rankings[0].id);
    }

    for (const featureId of Object.keys(observedEvidence)) {
      activeNodes.add(featureId);

      if (rankings.length > 0) {
        activeEdges.add(
          `${rankings[0].id}->${featureId}`
        );
      }
    }

    return {
      activeNodes:
        Array.from(activeNodes),

      activeEdges:
        Array.from(activeEdges)
    };
  }


  /**
   * Convenience method for UI code that wants the discretization
   * thresholds for a particular feature.
   */
  getThresholds(featureId) {
    return (
      this.model.thresholds[featureId] ??
      null
    );
  }


  /**
   * Exposes model metadata for the UI/report.
   */
  getModelMetadata() {
    return {
      ...this.model.metadata
    };
  }
}