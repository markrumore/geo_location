import argparse
import pandas as pd
from fuzzy_matching_module import FuzzyMatcher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_customers', type=str, required=True)
    parser.add_argument('--input_unmatched', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--matched_results', type=str, required=True)
    parser.add_argument('--keep_all', action='store_true', default=False)
    # Required for df1 (matched to)
    parser.add_argument('--zip_col1', type=str, default='POSTAL_CODE')
    parser.add_argument('--name_col1', type=str, default='CUSTOMER_DESC')
    # Optional for df2 (to match)
    parser.add_argument('--zip_col2', type=str, default=None)
    parser.add_argument('--name_col2', type=str, default=None)
    parser.add_argument('--address_col1', type=str, default='STREET_ADDRESS')
    parser.add_argument('--address_col2', type=str, default=None)
    parser.add_argument('--lat_col1', type=str, default='LATITUDE_COORDINATE')
    parser.add_argument('--long_col1', type=str, default='LONGITUDE_COORDINATE')
    parser.add_argument('--lat_col2', type=str, default=None)
    parser.add_argument('--long_col2', type=str, default=None)
    parser.add_argument('--threshold', type=int, default=95)
    parser.add_argument('--lat_long_tolerance', type=float, default=3)
    args = parser.parse_args()

    # Load data
    customers = pd.read_csv(args.input_customers)
    unmatched_df = pd.read_csv(args.input_unmatched)

    # Prepare kwargs for optional columns
    matcher_kwargs = dict(
        zip_col1=args.zip_col1,
        name_col1=args.name_col1,
        address_col1=args.address_col1,
        lat_col1=args.lat_col1,
        long_col1=args.long_col1,
        threshold=args.threshold,
        lat_long_tolerance=args.lat_long_tolerance
    )
    # Only add optional df2 columns if provided
    if args.zip_col2:
        matcher_kwargs['zip_col2'] = args.zip_col2
    if args.name_col2:
        matcher_kwargs['name_col2'] = args.name_col2
    if args.address_col2:
        matcher_kwargs['address_col2'] = args.address_col2
    if args.lat_col2:
        matcher_kwargs['lat_col2'] = args.lat_col2
    if args.long_col2:
        matcher_kwargs['long_col2'] = args.long_col2

    # Run fuzzy matching
    matcher = FuzzyMatcher(
        customers, unmatched_df,
        **matcher_kwargs
    )
    result = matcher.match(keep_all=args.keep_all)
    print(result.head())
    result.to_csv(args.matched_results, index=False)

if __name__ == "__main__":
    main()
