import pandas as pd
from rapidfuzz import process, fuzz
import string
import logging

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

class FuzzyMatcher:
    def __init__(self, df1, df2, zip_col1, zip_col2, name_col1, name_col2, 
                 address_col1=None, address_col2=None, lat_col1=None, long_col1=None, 
                 lat_col2=None, long_col2=None, threshold=75, lat_long_tolerance=0.01):
        """
        Initialize the FuzzyMatcher class with dataframes and column configurations.
        """
        self.df1 = df1
        self.df2 = df2
        self.zip_col1 = zip_col1
        self.zip_col2 = zip_col2
        self.name_col1 = name_col1
        self.name_col2 = name_col2
        self.address_col1 = address_col1
        self.address_col2 = address_col2
        self.lat_col1 = lat_col1
        self.long_col1 = long_col1
        self.lat_col2 = lat_col2
        self.long_col2 = long_col2
        self.threshold = threshold
        self.lat_long_tolerance = lat_long_tolerance

    @staticmethod
    def clean_zip_code(zip_code):
        """
        Clean the zip code by removing non-numeric characters and ensuring it is 5 digits long.
        """
        if pd.isna(zip_code):
            return None
        cleaned = ''.join(filter(str.isdigit, str(zip_code)))
        return cleaned.zfill(5)[:5]

    def zip_code_cleaner(self):
        """
        Clean zip code columns in both dataframes.
        """
        self.df1[self.zip_col1] = self.df1[self.zip_col1].apply(self.clean_zip_code)
        self.df2[self.zip_col2] = self.df2[self.zip_col2].apply(self.clean_zip_code)

    @staticmethod
    def clean_customer_name(name):
        """
        Clean customer names by removing punctuation and converting to lowercase.
        """
        if pd.isna(name):
            return None
        name = str(name)
        # Convert to string and remove punctuation
        return name.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    
    def customer_name_cleaner(self):
        """
        Clean customer name columns in both dataframes.
        """
        self.df1[self.name_col1] = self.df1[self.name_col1].apply(self.clean_customer_name)
        self.df2[self.name_col2] = self.df2[self.name_col2].apply(self.clean_customer_name)
    
    @staticmethod
    def clean_lat_long(lat, long, decimal_places=5):
        """
        Clean latitude and longitude values by rounding to a specified number of decimal places.

        Parameters:
        lat (float): Latitude value.
        long (float): Longitude value.
        decimal_places (int): Number of decimal places to round to (default is 5).

        Returns:
        tuple: Cleaned (lat, long) values rounded to the specified precision, or (None, None) if invalid.
        """
        try:
            lat = float(lat)
            long = float(long)
        except (ValueError, TypeError):
            return None, None

        # Round latitude and longitude to the specified number of decimal places
        lat = round(lat, decimal_places)
        long = round(long, decimal_places)

        return lat, long

    @staticmethod
    def get_decimal_places(value):
        """
        Get the number of decimal places in a float value.

        Parameters:
        value (float): The value to check.

        Returns:
        int: The number of decimal places, or 0 if the value is an integer or invalid.
        """
        try:
            value = float(value)
            decimal_part = str(value).split(".")[1]
            return len(decimal_part)
        except (ValueError, IndexError):
            return 0


    def lat_long_cleaner(self):
        """
        Clean latitude and longitude columns in both dataframes by rounding to a consistent decimal precision.
        """
        if not (self.lat_col1 and self.long_col1 and self.lat_col2 and self.long_col2):
            return

        # Apply cleaning to df1
        self.df1[[self.lat_col1, self.long_col1]] = self.df1[[self.lat_col1, self.long_col1]].apply(
            lambda row: self.clean_lat_long(row[self.lat_col1], row[self.long_col1], decimal_places=int(self.lat_long_tolerance)),
            axis=1, result_type='expand'
        )

        # Apply cleaning to df2
        self.df2[[self.lat_col2, self.long_col2]] = self.df2[[self.lat_col2, self.long_col2]].apply(
            lambda row: self.clean_lat_long(row[self.lat_col2], row[self.long_col2], decimal_places=int(self.lat_long_tolerance)),
            axis=1, result_type='expand'
        )

    @staticmethod
    def fuzzy_match(df1, df2, key1, key2, threshold=95):
        """
        Perform fuzzy matching between two DataFrame columns and return the best match for each row in df2.

        Parameters:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        key1 (str): The column name in df1 to match.
        key2 (str): The column name in df2 to match.
        threshold (int): The minimum score for a match to be considered valid.

        Returns:
        pd.DataFrame: A DataFrame containing the best match for each row in df2.
        """
        s = df1[key1].tolist()

        # Apply fuzzy matching
        match_results = []
        for idx, value in df2[key2].items():
            if pd.notna(value):
                # Compare the current value in df2 against all values in df1
                best_match = process.extractOne(value, s, scorer=fuzz.ratio)
                if best_match and best_match[1] >= threshold:
                    # Find the corresponding customer_id from df1
                    customer_id = df1.loc[df1[key1] == best_match[0], 'CUSTOMER_ID'].iloc[0]
                    match_results.append({
                        'df2_index': idx,
                        'best_match': best_match[0],  # Best match from df1
                        'match_score': best_match[1],  # Match score
                        'customer_id': customer_id,   # Corresponding customer_id
                        'is_matched': True
                    })
                else:
                    match_results.append({
                        'df2_index': idx,
                        'best_match': None,
                        'match_score': None,
                        'customer_id': None,
                        'is_matched': False
                    })
            else:
                match_results.append({
                    'df2_index': idx,
                    'best_match': None,
                    'match_score': None,
                    'customer_id': None,
                    'is_matched': False
                })

        # Convert match results to a DataFrame
        match_df = pd.DataFrame(match_results)

        return match_df

    def address_cleaner(self):
        """
        Clean address columns in both dataframes.
        """
        if self.address_col1 and self.address_col2:
            self.df1[self.address_col1] = self.df1[self.address_col1].fillna('')
            self.df2[self.address_col2] = self.df2[self.address_col2].fillna('')

            # Remove punctuation and convert to lowercase
            self.df1[self.address_col1] = self.df1[self.address_col1].str.translate(str.maketrans('', '', string.punctuation)).str.lower()
            self.df2[self.address_col2] = self.df2[self.address_col2].str.translate(str.maketrans('', '', string.punctuation)).str.lower()
            # Remove leading and trailing whitespace
            self.df1[self.address_col1] = self.df1[self.address_col1].str.strip()
            self.df2[self.address_col2] = self.df2[self.address_col2].str.strip()
            # Only log once for address cleaning
            logging.warning(f"Address columns '{self.address_col1}' and '{self.address_col2}' cleaned.")
        else:
            logging.warning("Address columns not provided. Skipping address cleaning.")

    def process(self):
        """
        Process the dataframes by cleaning zip codes, customer names, and optionally latitude/longitude.
        """
        logging.warning("Starting fuzzy matching preprocessing...")
        if self.zip_col1 and self.zip_col2:
            self.zip_code_cleaner()
        if self.name_col1 and self.name_col2:
            self.customer_name_cleaner()
        if self.lat_col1 and self.long_col1 and self.lat_col2 and self.long_col2:
            self.lat_long_cleaner()
        if self.address_col1 and self.address_col2:
            self.address_cleaner()
        logging.warning("Preprocessing complete.")

    def match(self, keep_all=False):
        """
        Perform optimized fuzzy matching:
        1. Match by postal code.
        2. Check for exact latitude/longitude matches and run fuzzy matcher at a high threshold on customer names.
        3. Handle remaining matches by combining address and customer description columns.
        4. Run remaining unmatched records through fuzzy matcher on customer names.

        Parameters:
        keep_all (bool): If True, return all rows from df2 with match info (default: False, only matched rows).

        Returns:
        pd.DataFrame: Dataframe with matches for each entry.
        """
        logging.warning("Starting fuzzy matching process...")
        # Step 1: Clean and prepare the dataframes
        self.process()

        # Add a column to track matched rows
        self.df2['is_matched'] = False

        # Initialize a list to store results
        result_dfs = []

        # Step 2: Match by postal code
        postal_codes = self.df1[self.zip_col1].unique()
        # Use tqdm for progress bar if available
        if tqdm:
            postal_iter = tqdm(postal_codes, desc="Matching by postal code")
        else:
            postal_iter = postal_codes

        for postal_code in postal_iter:
            df1_subset = self.df1[self.df1[self.zip_col1] == postal_code]
            df2_subset = self.df2[(self.df2[self.zip_col2] == postal_code) & (~self.df2['is_matched'])]

            if df2_subset.empty:
                continue

            # Step 3: Exact Latitude/Longitude Matches
            if self.lat_col1 and self.long_col1 and self.lat_col2 and self.long_col2:
                for (lat, long), df1_latlong_group in df1_subset.groupby([self.lat_col1, self.long_col1]):

                    df2_latlong_group = df2_subset[
                        (df2_subset[self.lat_col2] == lat) & (df2_subset[self.long_col2] == long)
                    ]

                    if df2_latlong_group.empty:
                        continue

                    # Perform fuzzy matching on customer names for exact matches
                    exact_matches_result = self.fuzzy_match(
                        df1_latlong_group, df2_latlong_group,
                        self.name_col1, self.name_col2,
                        threshold=80
                    )
                    exact_matches_result['match_type'] = 'lat-long'

                    # if keep_all keep all rows if not only keep matched rows
                    if not keep_all:
                        exact_matches_result = exact_matches_result[exact_matches_result['is_matched']]

                    result_dfs.append(exact_matches_result)

                    # Mark matched rows in df2 using df2_index
                    matched_ids = exact_matches_result['df2_index'][exact_matches_result['is_matched']].tolist()
                    self.df2.loc[matched_ids, 'is_matched'] = True

            # Step 4: Address and Customer Name Matching
            if self.address_col1 and self.address_col2:
                # Filter df2_subset to include only unmatched rows
                df2_subset = self.df2[(~self.df2['is_matched'])]

                # Create combined address and customer description columns for fuzzy matching
                df1_subset['address_customer_desc'] = (
                    df1_subset[self.name_col1].fillna('') + ' ' + df1_subset[self.address_col1].fillna('')
                )
                df2_subset['address_customer_desc'] = (
                    df2_subset[self.name_col2].fillna('') + ' ' + df2_subset[self.address_col2].fillna('')
                )

                # Perform fuzzy matching
                address_matches_result = self.fuzzy_match(
                    df1_subset, df2_subset, 'address_customer_desc', 'address_customer_desc', threshold=85
                )
                address_matches_result['match_type'] = 'address-zip'

                # if keep_all keep all rows if not only keep matched rows
                if not keep_all:
                    address_matches_result = address_matches_result[address_matches_result['is_matched']]
                
                # Only keep matched rows
                address_matches_result = address_matches_result[address_matches_result['is_matched']]
                result_dfs.append(address_matches_result)

                # Mark matched rows in df2 using df2_index
                matched_ids = address_matches_result['df2_index'][address_matches_result['is_matched']].tolist()
                self.df2.loc[matched_ids, 'is_matched'] = True

        # Combine all results into a single dataframe
        if result_dfs:
            final_result = pd.concat(result_dfs, ignore_index=True)
        else:
            final_result = pd.DataFrame()

        # Ensure customer_id and df2_index are included in the final output
        expected_cols = ['df2_index', 'best_match', 'match_score', 'customer_id', 'is_matched', 'match_type']
        if not final_result.empty:
            final_result = final_result[expected_cols]
        else:
            final_result = pd.DataFrame(columns=expected_cols)

        # Merge final_result with df2 to retain all rows from df2
        merged_result = self.df2.merge(final_result, how='left', left_index=True, right_on='df2_index')

        logging.warning(f"Fuzzy matching complete. Returning {len(merged_result) if keep_all else merged_result['customer_id'].notna().sum()} matched rows.")

        if keep_all:
            return merged_result
        else:
            # Only keep rows with a match (customer_id not null)
            return merged_result[merged_result['customer_id'].notna()]
