from __future__ import annotations

import math
from numbers import Real
from typing import Dict

from maths.probability.core.m00_probability_foundations import SampleSpace, Event, ProbabilityMeasure, ConditionalProbability, PROBABILITY_TOLERANCE

# Bayes' Theorem: P(H|E) = (P(E|H) * P(H)) / P(E)
class BayesTheorem:
    def __init__(self, prob_measure: ProbabilityMeasure) -> None:
        self.prob_measure = prob_measure
        self.cond_prob = ConditionalProbability(prob_measure)

    # Computes P(H|E) using instantiated Event and ProbabilityMeasure objects
    def compute_from_events(self, hypothesis: Event, evidence: Event) -> float:
        """Return P(H|E) directly; a zero-prior H correctly has posterior zero."""
        p_evidence = self.prob_measure(evidence)
        if p_evidence == 0.0:
            raise ValueError("P(Evidence) is 0; posterior is undefined.")
        p_joint = self.prob_measure(hypothesis & evidence)
        return p_joint / p_evidence      # Bayes Formula: (P(E|H) * P(H)) / P(E)

    # Computes P(H_k|E) using Law of Total Probability:
    # P(H_k|E) = (P(E|H_k) * P(H_k)) / Σ (P(E|H_i) * P(H_i))
    @staticmethod
    def compute_from_partition(priors: Dict[str, float], likelihoods: Dict[str, float], target_hypothesis: str) -> float:
        if set(priors) != set(likelihoods):
            raise ValueError("Keys in priors and likelihoods must match exactly.")
        if target_hypothesis not in priors:
            raise ValueError(f"Target hypothesis '{target_hypothesis}' not in priors.")

        clean_priors: dict[str, float] = {}
        clean_likelihoods: dict[str, float] = {}
        for key, raw in priors.items():
            if not isinstance(raw, Real):
                raise ValueError("Priors must be finite real numbers in [0, 1].")
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Priors must be finite real numbers in [0, 1].")
            clean_priors[key] = value

        for key, raw in likelihoods.items():
            if not isinstance(raw, Real):
                raise ValueError("Likelihoods must be finite real numbers in [0, 1].")
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Likelihoods must be finite real numbers in [0, 1].")
            clean_likelihoods[key] = value

        if abs(math.fsum(clean_priors.values()) - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError("Prior probabilities must sum to 1.0.")

        # Normalize likelihoods before multiplying by tiny priors. This also makes
        # common likelihood rescaling numerically harmless while information remains
        # representable in the input floats.
        scale = max(clean_likelihoods.values(), default=0.0)
        if scale == 0.0:
            raise ValueError("Total P(Evidence) is 0; posterior is undefined.")
        scaled_likelihoods = {key: value / scale for key, value in clean_likelihoods.items()}
        terms = {key: clean_priors[key] * scaled_likelihoods[key] for key in clean_priors}
        evidence_mass = math.fsum(terms.values())
        if evidence_mass == 0.0:
            raise ValueError("Total P(Evidence) is 0; posterior is undefined.")
        return terms[target_hypothesis] / evidence_mass

if __name__ == "__main__":
    print("Method 1: Event Driven Bayes (Urn Problem)")
    
    # Setup sample space for choosing an urn (Urn 1 or 2) and drawing a color (Red or Blue)
    sample_space = SampleSpace({
        ('Urn1', 'Red'), ('Urn1', 'Blue'),
        ('Urn2', 'Red'), ('Urn2', 'Blue')
    })

    # Outcome weights based on P(Urn) * P(Color | Urn)
    weights = {
        ('Urn1', 'Red'):  0.5 * 0.75,  # P(Urn1)=0.5, P(Red|Urn1)=0.75 -> 0.375
        ('Urn1', 'Blue'): 0.5 * 0.25,  # P(Urn1)=0.5, P(Blue|Urn1)=0.25 -> 0.125
        ('Urn2', 'Red'):  0.5 * 0.25,  # P(Urn2)=0.5, P(Red|Urn2)=0.25 -> 0.125
        ('Urn2', 'Blue'): 0.5 * 0.75   # P(Urn2)=0.5, P(Blue|Urn2)=0.75 -> 0.375
    }
    
    pm = ProbabilityMeasure(sample_space, weights)

    # Define Hypothesis event (Urn 1 selected) and Evidence event (Red ball drawn)
    hypothesis_urn1 = Event("Urn 1", {('Urn1', 'Red'), ('Urn1', 'Blue')}, sample_space)
    evidence_red    = Event("Red Ball Drawn", {('Urn1', 'Red'), ('Urn2', 'Red')}, sample_space)

    # Calculate P(Urn 1 | Red Ball)
    bayes = BayesTheorem(pm)
    p_urn1_given_red = bayes.compute_from_events(hypothesis_urn1, evidence_red)
    
    print(f"P(Urn 1) = {pm.P(hypothesis_urn1):.2f}")
    print(f"P(Red Ball) = {pm.P(evidence_red):.2f}")
    print(f"P(Urn 1 | Red Ball) = {p_urn1_given_red:.2f}\n")

    print("Method 2: Partition Driven Bayes (Medical Diagnostic Test)")
    
    # Population priors: 1% has disease, 99% is healthy
    priors = {
        "Disease": 0.01,
        "Healthy": 0.99
    }

    # Likelihoods: Test sensitivity is 95%, false positive rate is 5%
    likelihoods = {
        "Disease": 0.95,  # P(Test+ | Disease)
        "Healthy": 0.05   # P(Test+ | Healthy)
    }

    # Calculate P(Disease | Test+)
    p_disease_given_positive = BayesTheorem.compute_from_partition(
        priors=priors,
        likelihoods=likelihoods,
        target_hypothesis="Disease"
    )

    print(f"Prior P(Disease): {priors['Disease'] * 100:.1f}%")
    print(f"Likelihood P(Positive | Disease): {likelihoods['Disease'] * 100:.1f}%")
    print(f"Likelihood P(Positive | Healthy): {likelihoods['Healthy'] * 100:.1f}%")
    print(f"Posterior P(Disease | Positive Test) = {p_disease_given_positive:.4f} "
          f"({p_disease_given_positive * 100:.2f}%)")


    