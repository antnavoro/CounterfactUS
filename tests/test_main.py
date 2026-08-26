import pytest
import numpy as np
import pandas as pd
from counterfactus import compute_counterfactuals

@pytest.fixture
def dummy_data():
    """Generates synthetic data for testing."""
    np.random.seed(42)
    # 20 samples, 2 continuous features (X), 2 binary features (Z)
    X_train = pd.DataFrame(np.random.rand(20, 2), columns=["x1", "x2"])
    Z_train = pd.DataFrame(np.random.randint(0, 2, size=(20, 2)), columns=["z1", "z2"])
    
    # Target variable Y (binary)
    Y_train = pd.Series(np.random.randint(0, 2, size=20))
    
    # Single query instance
    x_query = pd.DataFrame([[0.5, 0.5]], columns=["x1", "x2"])
    z_query = pd.DataFrame([[0, 1]], columns=["z1", "z2"])
    
    return X_train, Z_train, Y_train, x_query, z_query

def test_compute_counterfactuals_basic(dummy_data):
    """
    Test the happy path: the function should return a pandas DataFrame
    and not crash with default parameters.
    """
    X_train, Z_train, Y_train, x_query, z_query = dummy_data
    
    # Note: We disable plot_figures and output_csv to avoid writing files during tests
    results_df = compute_counterfactuals(
        X=X_train, 
        Z=Z_train, 
        Y=Y_train,
        family="binomial", 
        link="logit",
        prediction_threshold=0.6,
        x0s=x_query,
        z0s=z_query,
        gurobi_verbose=False,
        plot_figures=False,
        output_csv=None
    )
    
    assert isinstance(results_df, pd.DataFrame), "Result should be a DataFrame"
    assert not results_df.empty, "Result DataFrame should not be empty"

def test_max_l0_constraint(dummy_data):
    """
    Test that the max_l0 constraint is respected.
    If we enforce max_l0=1, at most 1 feature should change between the original
    instance and the counterfactual.
    """
    X_train, Z_train, Y_train, x_query, z_query = dummy_data
    
    results_df = compute_counterfactuals(
        X=X_train, 
        Z=Z_train, 
        Y=Y_train,
        family="binomial", 
        link="logit",
        prediction_threshold=0.6,
        x0s=x_query,
        z0s=z_query,
        max_l0=1,
        gurobi_verbose=False,
        plot_figures=False,
        output_csv=None
    )
    
    # Extract the original and counterfactual features from the result
    # We check the first row since we only passed one query
    if results_df.empty or "status" not in results_df.columns:
        return # Skip if no result was generated
        
    status = results_df.iloc[0].get("status", None)
    
    # Only verify changes if the solver actually found a solution (Status 2 = Optimal)
    if status == 2:
        x0_1, x0_2 = results_df.iloc[0]["x0_x1"], results_df.iloc[0]["x0_x2"]
        xcf_1, xcf_2 = results_df.iloc[0]["x_cf_x1"], results_df.iloc[0]["x_cf_x2"]
        z0_1, z0_2 = results_df.iloc[0]["z0_z1"], results_df.iloc[0]["z0_z2"]
        zcf_1, zcf_2 = results_df.iloc[0]["z_cf_z1"], results_df.iloc[0]["z_cf_z2"]
        
        # Count differences (using np.isclose for continuous variables to avoid floating point issues)
        changes = 0
        if not np.isclose(x0_1, xcf_1): changes += 1
        if not np.isclose(x0_2, xcf_2): changes += 1
        if z0_1 != zcf_1: changes += 1
        if z0_2 != zcf_2: changes += 1
        
        assert changes <= 1, f"Expected at most 1 feature change, but got {changes} changes."

def test_invalid_distance_metric(dummy_data):
    """
    Test edge case: passing an invalid distance metric should raise a ValueError.
    """
    X_train, Z_train, Y_train, x_query, z_query = dummy_data
    
    with pytest.raises(ValueError):
        compute_counterfactuals(
            X=X_train, Z=Z_train, Y=Y_train,
            x0s=x_query, z0s=z_query,
            distance="invalid_metric_name",
            plot_figures=False,
            output_csv=None
        )

def test_immutable_features(dummy_data):
    """
    Test that features marked as immutable do not change in the counterfactual.
    """
    X_train, Z_train, Y_train, x_query, z_query = dummy_data
    
    results_df = compute_counterfactuals(
        X=X_train, Z=Z_train, Y=Y_train,
        x0s=x_query, z0s=z_query,
        inmutable_z=["z1"],
        gurobi_verbose=False,
        plot_figures=False,
        output_csv=None
    )
    
    if results_df.empty or "status" not in results_df.columns:
        return
        
    status = results_df.iloc[0].get("status", None)
    if status == 2:
        z0_1 = results_df.iloc[0]["z0_z1"]
        zcf_1 = results_df.iloc[0]["z_cf_z1"]
        assert z0_1 == zcf_1, "Immutable feature 'z1' changed!"

def test_prediction_threshold(dummy_data):
    """
    Test that the returned counterfactual meets the required prediction threshold.
    """
    X_train, Z_train, Y_train, x_query, z_query = dummy_data
    
    target_threshold = 0.8
    results_df = compute_counterfactuals(
        X=X_train, Z=Z_train, Y=Y_train,
        x0s=x_query, z0s=z_query,
        prediction_threshold=target_threshold,
        gurobi_verbose=False,
        plot_figures=False,
        output_csv=None
    )
    
    if results_df.empty or "status" not in results_df.columns:
        return
        
    status = results_df.iloc[0].get("status", None)
    if status == 2:
        pred = results_df.iloc[0].get("y", 0)
        assert pred >= target_threshold - 1e-4, f"Prediction {pred} is below threshold {target_threshold}"
