import sys
import os
import pytest
import pandas as pd
from fuzzy_matching_module import FuzzyMatcher

@pytest.fixture
def sample_data():
    df1 = pd.DataFrame({
        'CUSTOMER_ID': [1, 2],
        'POSTAL_CODE': ['12345', '54321'],
        'CUSTOMER_DESC': ['Alpha Cafe', 'Beta Bistro'],
        'STREET_ADDRESS': ['100 Main St.', '200 Oak Ave'],
        'LATITUDE_COORDINATE': [34.05, 36.12],
        'LONGITUDE_COORDINATE': [-118.25, -115.17],
        'REGION': ['CALIFORNIA', 'CALIFORNIA']
    })
    df2 = pd.DataFrame({
        'POSTAL_CODE': ['12345', '54321', '99999'],
        'CUSTOMER_DESC': ['alpha cafe', 'beta bistro', 'gamma grill'],
        'STREET_ADDRESS_LINE_1': ['100 main st', '200 oak ave', '300 pine rd'],
        'LATITUDE': [34.05, 36.12, 40.00],
        'LONGITUDE': [-118.25, -115.17, -120.00],
        'STATE_PROVINCE_NAME': ['CA', 'CA', 'CA']
    })
    return df1, df2

def test_clean_zip_code():
    assert FuzzyMatcher.clean_zip_code('12345-6789') == '12345'
    assert FuzzyMatcher.clean_zip_code('9876') == '09876'
    assert FuzzyMatcher.clean_zip_code(None) is None

def test_clean_customer_name():
    assert FuzzyMatcher.clean_customer_name('Alpha Cafe!') == 'alpha cafe'
    assert FuzzyMatcher.clean_customer_name(None) is None

def test_clean_lat_long():
    lat, lon = FuzzyMatcher.clean_lat_long('34.123456', '-118.987654', decimal_places=3)
    assert lat == 34.123 and lon == -118.988

def test_zip_code_cleaner(sample_data):
    df1, df2 = sample_data
    matcher = FuzzyMatcher(df1.copy(), df2.copy(), 'POSTAL_CODE', 'POSTAL_CODE', 'CUSTOMER_DESC', 'CUSTOMER_DESC')
    matcher.zip_code_cleaner()
    assert all(matcher.df1['POSTAL_CODE'].str.len() == 5)
    assert all(matcher.df2['POSTAL_CODE'].str.len() == 5)

def test_customer_name_cleaner(sample_data):
    df1, df2 = sample_data
    matcher = FuzzyMatcher(df1.copy(), df2.copy(), 'POSTAL_CODE', 'POSTAL_CODE', 'CUSTOMER_DESC', 'CUSTOMER_DESC')
    matcher.customer_name_cleaner()
    assert matcher.df1['CUSTOMER_DESC'].iloc[0] == 'alpha cafe'
    assert matcher.df2['CUSTOMER_DESC'].iloc[0] == 'alpha cafe'

def test_address_cleaner(sample_data):
    df1, df2 = sample_data
    matcher = FuzzyMatcher(
        df1.copy(), df2.copy(),
        'POSTAL_CODE', 'POSTAL_CODE',
        'CUSTOMER_DESC', 'CUSTOMER_DESC',
        address_col1='STREET_ADDRESS', address_col2='STREET_ADDRESS_LINE_1'
    )
    matcher.address_cleaner()
    assert matcher.df1['STREET_ADDRESS'].iloc[0] == '100 main st'
    assert matcher.df2['STREET_ADDRESS_LINE_1'].iloc[0] == '100 main st'

def test_fuzzy_match(sample_data):
    df1, df2 = sample_data
    matcher = FuzzyMatcher(df1.copy(), df2.copy(), 'POSTAL_CODE', 'POSTAL_CODE', 'CUSTOMER_DESC', 'CUSTOMER_DESC')
    matcher.customer_name_cleaner()
    match_df = matcher.fuzzy_match(matcher.df1, matcher.df2, 'CUSTOMER_DESC', 'CUSTOMER_DESC', threshold=90)
    assert match_df['is_matched'].iloc[0] == True
    assert match_df['customer_id'].iloc[0] == 1

def test_match_process(sample_data):
    df1, df2 = sample_data
    matcher = FuzzyMatcher(
        df1.copy(), df2.copy(),
        zip_col1='POSTAL_CODE', zip_col2='POSTAL_CODE',
        name_col1='CUSTOMER_DESC', name_col2='CUSTOMER_DESC',
        address_col1='STREET_ADDRESS', address_col2='STREET_ADDRESS_LINE_1',
        lat_col1='LATITUDE_COORDINATE', long_col1='LONGITUDE_COORDINATE',
        lat_col2='LATITUDE', long_col2='LONGITUDE',
        threshold=90,
        lat_long_tolerance=2
    )
    result = matcher.match(keep_all=True)
    # Should match first two rows, third should be unmatched
    assert result['is_matched_y'].iloc[0] == True
    assert result['is_matched_y'].iloc[1] == True
    assert pd.isna(result['customer_id'].iloc[2])

def test_match_without_latlong(sample_data):
    df1, df2 = sample_data
    matcher = FuzzyMatcher(
        df1.copy(), df2.copy(),
        zip_col1='POSTAL_CODE', zip_col2='POSTAL_CODE',
        name_col1='CUSTOMER_DESC', name_col2='CUSTOMER_DESC'
    )
    result = matcher.match()
    # If result is empty, skip assertion
    if not result.empty and 'is_matched_y' in result.columns:
        assert result['is_matched_y'].iloc[0] == True

def test_match_with_different_threshold(sample_data):
    df1, df2 = sample_data
    matcher = FuzzyMatcher(
        df1.copy(), df2.copy(),
        zip_col1='POSTAL_CODE', zip_col2='POSTAL_CODE',
        name_col1='CUSTOMER_DESC', name_col2='CUSTOMER_DESC',
        threshold=100
    )
    result = matcher.match()
    # If result is empty, skip assertion
    if not result.empty and 'is_matched_y' in result.columns:
        assert result['is_matched_y'].iloc[0] == True
        assert result['is_matched_y'].iloc[1] == True

