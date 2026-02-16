import pytest

from fast_lisa_subtraction import GalacticBinaryPopulation


def test_galactic_binary_population():
    
    # Create a population with 10^3 sources
    GB_population = GalacticBinaryPopulation()

    # Sample the population 
    N = int(1e3)
    population_samples = GB_population.sample(N, copula=True, kind='gaussian', rho=0.9)

    # Convert to dataframe
    df = population_samples.dataframe()

    # Check that the dataframe has the expected columns
    expected_columns = ['Frequency', 'FrequencyDerivative', 'Amplitude', 'InitialPhase',
       'Inclination', 'Polarization', 'EclipticLongitude', 'EclipticLatitude']

    assert all(col in df.columns for col in expected_columns), "DataFrame is missing expected columns"
    
    # Check that the dataframe has N rows
    assert len(df) == N, f"DataFrame has {len(df)} rows, expected {N}"


if __name__ == "__main__":
    pytest.main([__file__])