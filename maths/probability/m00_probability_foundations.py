from typing import Any, Dict, Optional

# Sample Space: Set of all possible outcomes of a random experiment
class SampleSpace:
    def __init__(self, outcomes):
        self.outcomes = set(outcomes)

    def __repr__(self):
        return f"SampleSpace({self.outcomes})"

    def __contains__(self, item: Any) -> bool:
        return item in self.outcomes    

# Event: A subset of outcomes within a defined sample space 
class Event:
    def __init__(self, name: str, outcomes: set, sample_space: SampleSpace):
        outcomes_set = set(outcomes)
        if not outcomes_set.issubset(sample_space.outcomes):
            raise ValueError("All event outcomes must belong to the sample space.")
        self.name = name
        self.outcomes = outcomes_set
        self.sample_space = sample_space

    def __repr__(self):
        return F"Event('{self.name}', outcomes = {self.outcomes})"    

# Set operations: Handles union, intersection, compliment and difference
class SetOperations:
    @staticmethod
    def union(e1: Event, e2: Event, name: Optional[str] = None) -> Event:
        if e1.sample_space != e2.sample_space:
            raise ValueError("Event must share exact same sample sapce.")
        event_name = name or f"({e1.name} U {e2.name})"
        return Event(event_name, e1.outcomes | e2.outcomes, e1.sample_space)

    @staticmethod
    def intersection(e1: Event, e2: Event, name: Optional[str] = None) -> Event:
        if e1.sample_space != e2.sample_space:
            raise ValueError("Event must share exact same sample space.")
        event_name = name or F"({e1.name} n {e2.name})"
        return Event(event_name, e1.outcomes & e2.outcomes, e1.sample_space)

    @staticmethod
    def complement(e: Event, name: Optional[str] = None) -> Event:
        event_name = name or f"({e.name})"
        comp_outcome = e.sample_space.outcomes - e.outcomes
        return Event(event_name, comp_outcome, e.sample_space)

    @staticmethod
    def difference(e1: Event, e2: Event, name: Optional[str] = None) -> Event:
        if e1.sample_space != e2.sample_space:
            raise ValueError("Event must share exact same sample space.")
        event_name = name or f"({e1.name} - {e2.name})"
        return Event(event_name, e1.outcomes - e2.outcomes, e1.sample_space)

# Probability measure: Assigns real number probabilities to events.
class ProbabilityMeasure:
    def __init__(self, sample_space: SampleSpace, weights: Optional[Dict[Any, float]] = None):
        self.sample_space = sample_space
        if weights is None:
            # deafult to uniform distribution
            n = len(sample_space.outcomes)
            self.weights = {out: 1.0 / n for out in sample_space.outcomes}
        else:
            if set(weights.keys()) != sample_space.outcomes:
                raise ValueError("Weights must cover every column in the sample space.")
            if not abs(sum(weights.values()) - 1.0) < 1e-6:
                raise ValueError("Sum of outcome of probabilities must be equal to one.")
            for p in weights.values():
                if p < 0:
                    raise ValueError("Probability cannot be negative.")
            self.weights = weights

    # Calculates P(E) by summing the probability weights of its outcomes.
    def P(self, event: Event) -> float:
        if event.sample_space != self.sample_space:
            raise ValueError("Event does not belongs to this probability measure's sample space.")
        return sum(self.weights[outcome] for outcome in event.outcomes)           

# Conditional Probability P(A|B): Calculates P(A ∩ B) / P(B).
class ConditionalProbability:
    def __init__(self, prob_measure: ProbabilityMeasure):
        self.prob_measure = prob_measure

    def compute(self, event_A: Event, event_B: Event) -> float:
        p_B = self.prob_measure.P(event_B)
        if p_B == 0:
            raise ValueError("Conditional probability P(A|B) is undefined when P(B) = 0.")
        intersection_event = SetOperations.intersection(event_A, event_B)
        p_A_and_B = self.prob_measure.P(intersection_event)
        return p_A_and_B / p_B    

if __name__ == "__main__":
    # Example: Rolling a 6 sided die.
    # 1. Instantiate Sample Space 
    omega = SampleSpace({1, 2, 3, 4, 5, 6})

    # 2. Define Events
    even_event = Event("Even Number", {2, 4, 6}, omega)
    greater_than_3 = Event("Greater Than 3", {4, 5, 6}, omega)

    # 3. Perform Set Operations
    even_and_gt3 = SetOperations.intersection(even_event, greater_than_3)
    even_or_gt3 = SetOperations.union(even_event, greater_than_3)
    not_even = SetOperations.complement(even_event)

    print(f"Intersection (Even & >3): {even_and_gt3.outcomes}")  # {4, 6}
    print(f"Union (Even or >3):        {even_or_gt3.outcomes}")   # {2, 4, 5, 6}
    print(f"Complement (not Even):     {not_even.outcomes}")       # {1, 3, 5}

    # 4. Measure Probability (Fair die with uniform probabilities)
    pm = ProbabilityMeasure(omega)
    print(f"P(Even) = {pm.P(even_event):.2f}")                    # 0.50
    print(f"P(Greater Than 3) = {pm.P(greater_than_3):.2f}")       # 0.50

    # 5. Compute Conditional Probability: P(Even | Greater Than 3)
    cond_prob = ConditionalProbability(pm)
    p_even_given_gt3 = cond_prob.compute(even_event, greater_than_3)
    print(f"P(Even | >3) = {p_even_given_gt3:.2f}")                # 0.67 (2 out of 3: {4,6} from {4,5,6})
