from sklearn.ensemble import RandomForestClassifier

def classifier_model( random_state: int =42 ) -> RandomForestClassifier:
    """
    Creates and returns a RandomForestClassifier model.

    Parameters:
    - random_state: Seed used by the random number generator for reproducibility.

    Returns:
    - An instance of RandomForestClassifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    return model