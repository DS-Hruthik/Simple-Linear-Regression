import os
import pandas as pd
import joblib
import pickle
from urllib.parse import quote
from sqlalchemy import create_engine

# Try to load preprocessing objects once (on app start)
_preprocessor = None
_impute_obj = None
_winsor_obj = None

# try single combined preprocessor
if os.path.exists('preprocessed'):
    try:
        _preprocessor = joblib.load('preprocessed')
    except Exception as e:
        _preprocessor = None

# try separate items if combined not available
if _preprocessor is None:
    if os.path.exists('imputation'):
        try:
            _impute_obj = joblib.load('imputation')
        except Exception as e:
            _impute_obj = None
    if os.path.exists('winzor'):
        try:
            _winsor_obj = joblib.load('winzor')
        except Exception as e:
            _winsor_obj = None

# load model
poly_model = pickle.load(open('poly_model.pkl', 'rb'))


def apply_preprocessing(df):
    """
    df: original dataframe uploaded by user (pandas.DataFrame)
    returns: preprocessed dataframe (pandas.DataFrame) ready for model.predict
    """
    # choose numeric columns (exclude object)
    numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found. Please upload data with numeric features.")
    X = df[numeric_cols]

    # If combined preprocessor exists, use it
    if _preprocessor is not None:
        arr = _preprocessor.transform(X)
        return pd.DataFrame(arr, columns=numeric_cols)

    # else apply impute then winsor (if available)
    arr = X.values
    if _impute_obj is not None:
        arr = _impute_obj.transform(arr)
    # winsorizer expects dataframe or array depending on how it was saved; handle both
    if _winsor_obj is not None:
        try:
            arr = _winsor_obj.transform(arr)
        except Exception:
            # if winsor expects dataframe
            arr = _winsor_obj.transform(pd.DataFrame(arr, columns=numeric_cols)).values

    return pd.DataFrame(arr, columns=numeric_cols)


def predict_AT(data, user, pw, db, table_name='mpg_predictions'):
    """
    - data: pandas DataFrame from uploaded file
    - DB creds: user, pw, db
    - returns: final DataFrame with predictions appended
    """
    if data is None or data.empty:
        raise ValueError("No input data provided")

    # apply preprocessing
    clean1 = apply_preprocessing(data)

    # model prediction (sklearn pipeline expects the same numeric columns shape)
    prediction = pd.DataFrame(poly_model.predict(clean1), columns=['Pred_AT'])

    final = pd.concat([prediction, data.reset_index(drop=True)], axis=1)

    # save to database
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
    final.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=1000)

    return final
