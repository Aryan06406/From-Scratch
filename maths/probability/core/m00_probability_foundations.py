from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional

PROBABILITY_TOLERANCE = 1e-9

# Sample Space
# Represents the sample space (Ω) of a random experiment.
# A sample space is the set of all possible outcomes.
class SampleSpace:
    def __init__(self, outcomes: Iterable[Any]):
        try:
            self.outcomes = set(outcomes)
        except TypeError as exc:
            raise TypeError("Sample space outcomes must be hashable.") from exc
        if not self.outcomes:
            raise ValueError("Sample space cannot be empty.")

    def __contains__(self, item: Any) -> bool:
        return item in self.outcomes

    def __len__(self) -> int:
        return len(self.outcomes)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, SampleSpace) and self.outcomes == other.outcomes)

    def __repr__(self) -> str:
        return f"SampleSpace({self.outcomes})"
    
# Event: An event is a subset of the sample space.
class Event:
    def __init__(self, name: str, outcomes: Iterable[Any], sample_space: SampleSpace):
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace.")
        outcomes = set(outcomes)
        if not outcomes.issubset(sample_space.outcomes):
            raise ValueError("All event outcomes must belong to the sample space.")
        self.name = name
        self.outcomes = outcomes
        self.sample_space = sample_space

    def __repr__(self) -> str:
        return f"Event('{self.name}', outcomes={self.outcomes})"

    # Event utilities
    # Returns True if the event contains no outcomes.
    def is_empty(self) -> bool:
        return not self.outcomes

    # Returns the number of outcomes in the event.
    def cardinality(self) -> int:
        return len(self.outcomes)

    # Checks if this event is a subset of another.
    def is_subset(self, other: "Event") -> bool:
        return self.outcomes.issubset(other.outcomes)

    # Checks if two events are disjoint. 
    def is_disjoint(self, other: "Event") -> bool:
        return self.outcomes.isdisjoint(other.outcomes)

    def _check_other(self, other: "Event") -> None:
        if not isinstance(other, Event) or self.sample_space != other.sample_space:
            raise ValueError("Events must belong to the same sample space.")

    # Operator overloading
    def __or__(self, other: "Event") -> "Event":
        return SetOperations.union(self, other)

    def __and__(self, other: "Event") -> "Event":
        return SetOperations.intersection(self, other)

    def __sub__(self, other: "Event") -> "Event":
        return SetOperations.difference(self, other)

    def __invert__(self) -> "Event":
        return SetOperations.complement(self)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Event) and self.outcomes == other.outcomes and self.sample_space == other.sample_space)

    def is_superset(self, other: "Event") -> bool:
        return self.outcomes.issuperset(other.outcomes)

    def contains(self, outcome: Any) -> bool:
        return outcome in self.outcomes

    def __contains__(self, outcome: Any) -> bool:
        return self.contains(outcome)

    def copy(self, name: Optional[str] = None) -> "Event":
        return Event(self.name if name is None else name, self.outcomes.copy(), self.sample_space)

    @classmethod
    def from_predicted(cls, name: str, sample_space: SampleSpace, predicate: Callable[[Any], bool]) -> "Event":
        return cls(name, (x for x in sample_space.outcomes if predicate(x)), sample_space)

# Set Operations: Implements common set operations on events. 
class SetOperations:
    @staticmethod
    def _same_space(e1: Event, e2: Event) -> None:
        if e1.sample_space != e2.sample_space:
            raise ValueError("Events must belong to the same sample space.")

    @staticmethod
    def union(e1: Event, e2: Event, name: Optional[str] = None) -> Event:
        SetOperations._same_space(e1, e2)
        return Event(name or f"({e1.name} U {e2.name})", e1.outcomes | e2.outcomes, e1.sample_space)

    @staticmethod
    def intersection(e1: Event, e2: Event, name: Optional[str] = None) -> Event:
        SetOperations._same_space(e1, e2)
        return Event(name or f"({e1.name} ∩ {e2.name})", e1.outcomes & e2.outcomes, e1.sample_space)

    @staticmethod
    def complement(event: Event, name: Optional[str] = None) -> Event:
        return Event(name or f"{event.name}ᶜ", event.sample_space.outcomes - event.outcomes, event.sample_space)

    @staticmethod
    def difference(e1: Event, e2: Event, name: Optional[str] = None) -> Event:
        SetOperations._same_space(e1, e2)
        return Event(name or f"({e1.name} - {e2.name})", e1.outcomes - e2.outcomes, e1.sample_space)

# Probability Measure: Assigns probabilities to outcomes in a sample space. Supports 
# both uniform and custom probability distributions.
class ProbabilityMeasure:
    """A probability measure over a finite sample space."""

    def __init__(self, sample_space: SampleSpace, weights: Optional[Dict[Any, float]] = None):
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace.")
        self.sample_space = sample_space

        if weights is None:
            probability = 1.0 / len(sample_space)
            self.weights = {outcome: probability for outcome in sample_space.outcomes}
            return

        if set(weights) != sample_space.outcomes:
            raise ValueError("Weights must be defined for every outcome and no others.")

        copied: Dict[Any, float] = {}
        for outcome, raw_probability in weights.items():
            try:
                probability = float(raw_probability)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Every probability weight must be a finite real number.") from exc
            if not math.isfinite(probability):
                raise ValueError("Probability weights must be finite.")
            if probability < 0.0 or probability > 1.0:
                raise ValueError("Every probability must lie in [0, 1].")
            copied[outcome] = probability

        if abs(math.fsum(copied.values()) - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError("Probabilities must sum to 1.")
        self.weights = copied

    # Allows probability measure to be called like: P(event)
    def __call__(self, event: Event) -> float:
        if not isinstance(event, Event) or event.sample_space != self.sample_space:
            raise ValueError("Event belongs to a different sample space.")
        return math.fsum(self.weights[outcome] for outcome in event.outcomes)

    # Alias for readability.
    def P(self, event: Event) -> float:
        return self(event)

# Conditional Probability: Computes conditional probabilities. P(A|B) = P(A∩B) / P(B)
class ConditionalProbability:
    def __init__(self, probability_measure: ProbabilityMeasure):
        self.probability_measure = probability_measure

    def compute(self, event_A: Event, event_B: Event) -> float:
        p_b = self.probability_measure(event_B)
        if p_b == 0.0:
            raise ValueError("Conditional probability is undefined because P(B) = 0.")
        return self.probability_measure(event_A & event_B) / p_b    

# Maps outcomes from a sample space to numeric values.
class RandomVariable:
    def __init__(self, sample_space: SampleSpace, mapping: Callable[[Any], float]):
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace.")
        self.sample_space = sample_space
        self.mapping = mapping

    def __call__(self, outcome: Any) -> float:
        if outcome not in self.sample_space:
            raise ValueError("Outcome not in sample space.")
        return self.mapping(outcome)  

    # Constructs Event {ω ∈ Ω : X(ω) == value}.
    def event_equals(self, value: float) -> Event:
        outcomes = {outcome for outcome in self.sample_space.outcomes if self.mapping(outcome) == value}
        return Event(f"X = {value}", outcomes, self.sample_space)  

    # Constructs Event {ω ∈ Ω : X(ω) <= value}.
    def event_less_than_or_equal(self, value: float) -> Event:
        outcomes = {outcome for outcome in self.sample_space.outcomes if self.mapping(outcome) <= value}
        return Event(f"X <= {value}", outcomes, self.sample_space)
    
# Example Usage
if __name__ == "__main__":

    # Sample Space
    omega = SampleSpace({1, 2, 3, 4, 5, 6})

    # Events
    even = Event("Even", {2, 4, 6}, omega)
    greater_than_3 = Event("Greater Than 3", {4, 5, 6}, omega)

    # Set Operations
    intersection = even & greater_than_3
    union = even | greater_than_3
    complement = ~even
    difference = greater_than_3 - even

    # Random Variable
    # X(ω) = ω² (square of the die outcome)
    X = RandomVariable(omega, lambda outcome: outcome ** 2)

    # Probability Measure 
    P = ProbabilityMeasure(omega)

    print("Random Variable X(ω) = ω²")
    for outcome in sorted(omega.outcomes):
        print(f"X({outcome}) = {X(outcome)}")

    print()

    # Events induced by the random variable
    X_equals_16 = X.event_equals(16)
    X_at_most_9 = X.event_less_than_or_equal(9)

    print(f"{X_equals_16.name}: {X_equals_16.outcomes}")
    print(f"{X_at_most_9.name}: {X_at_most_9.outcomes}")

    print()

    # Probabilities of the induced events
    print(f"P({X_equals_16.name}) = {P(X_equals_16):.2f}")
    print(f"P({X_at_most_9.name}) = {P(X_at_most_9):.2f}")

    # Other outputs
    print("Intersection :", intersection.outcomes)
    print("Union        :", union.outcomes)
    print("Complement   :", complement.outcomes)
    print("Difference   :", difference.outcomes)

    print()

    # Probability Measure
    print(f"P(Even) = {P(even):.2f}")
    print(f"P(Greater Than 3) = {P(greater_than_3):.2f}")
    print(f"P(Even ∩ >3) = {P(intersection):.2f}")

    print()

    # Conditional Probability
    CP = ConditionalProbability(P)

    print(
        f"P(Even | Greater Than 3) = "
        f"{CP.compute(even, greater_than_3):.2f}"
    )