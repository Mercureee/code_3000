import pandas as pd


def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.
    
    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    qids = ['age', 'zip3', 'gender']

    matches = pd.merge(anon_df, aux_df, on=qids, how='inner')

    unique_matches = matches.drop_duplicates(subset=qids, keep=False)

    unique_matches = unique_matches.rename(columns={'name': 'matched_name'})
    return unique_matches[['anon_id', 'matched_name']]


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0

    rate = len(matches_df) / len(anon_df)

    return float(rate)


anon_path = "mod06_data/anonymized.csv"
aux_path = "mod06_data/auxiliary.csv"

anon, aux = load_data(anon_path, aux_path)
anon.head(20)

matches = link_records(anon, aux)
matches.head()

rate = deanonymization_rate(matches, anon)
print(rate)