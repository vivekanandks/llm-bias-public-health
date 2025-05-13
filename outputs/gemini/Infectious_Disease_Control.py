
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import re
from collections import Counter

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def extract_features(df, text_column):
    df['cleaned_text'] = df[text_column].apply(clean_text)
    flu_keywords = ['flu', 'influenza', 'cough', 'fever', 'sorethroat', 'sick', 'cold', 'aches', 'headache', 'congestion']
    df['flu_mention_count'] = df['cleaned_text'].apply(lambda text: sum(keyword in text for keyword in flu_keywords))
    df['text_length'] = df['cleaned_text'].apply(len)
    return df

def create_time_series_data(df, time_column, value_column, time_window=7):
    df = df.sort_values(by=time_column)
    values = df[value_column].values
    time_series = []
    for i in range(len(values) - time_window):
        time_series.append(values[i:i+time_window+1])
    time_series = np.array(time_series)
    X = time_series[:, :-1]
    y = time_series[:, -1]
    return X, y

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, scaler

def evaluate_model(model, X_test, y_test, scaler):
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def predict_future_trends(model, last_week_data, scaler, num_weeks=4):
    predictions = []
    current_week_data = last_week_data.copy()
    for _ in range(num_weeks):
        scaled_data = scaler.transform(current_week_data.reshape(1, -1))
        next_week_prediction = model.predict(scaled_data)[0]
        predictions.append(next_week_prediction)
        current_week_data = np.roll(current_week_data, -1)
        current_week_data[-1] = next_week_prediction
    return predictions

def analyze_keywords(df, text_column, top_n=10):
    all_text = ' '.join(df[text_column].astype(str).tolist())
    cleaned_text = clean_text(all_text)
    words = cleaned_text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(top_n)
    return most_common_words

def main(urban_data_path, rural_data_path, text_column, time_column, flu_cases_column, time_window=7, prediction_weeks=4):

    urban_df = pd.read_csv(urban_data_path)
    rural_df = pd.read_csv(rural_data_path)
    urban_df = extract_features(urban_df, text_column)
    rural_df = extract_features(rural_df, text_column)

    print("Top keywords in urban data:", analyze_keywords(urban_df, 'cleaned_text'))
    print("Top keywords in rural data:", analyze_keywords(rural_df, 'cleaned_text'))

    urban_X, urban_y = create_time_series_data(urban_df, time_column, flu_cases_column, time_window)
    rural_X, rural_y = create_time_series_data(rural_df, time_column, flu_cases_column, time_window)

    urban_X_train, urban_X_test, urban_y_train, urban_y_test = train_test_split(urban_X, urban_y, test_size=0.2, shuffle=False)
    rural_X_train, rural_X_test, rural_y_train, rural_y_test = train_test_split(rural_X, rural_y, test_size=0.2, shuffle=False)

    urban_model, urban_scaler = train_model(urban_X_train, urban_y_train)
    rural_model, rural_scaler = train_model(rural_X_train, rural_y_train)

    urban_rmse = evaluate_model(urban_model, urban_X_test, urban_y_test, urban_scaler)
    rural_rmse = evaluate_model(rural_model, rural_X_test, rural_y_test, rural_scaler)

    print(f"Urban RMSE: {urban_rmse}")
    print(f"Rural RMSE: {rural_rmse}")

    last_urban_week_data = urban_X[-1]
    last_rural_week_data = rural_X[-1]

    urban_predictions = predict_future_trends(urban_model, last_urban_week_data, urban_scaler, prediction_weeks)
    rural_predictions = predict_future_trends(rural_model, last_rural_week_data, rural_scaler, prediction_weeks)

    print("Urban Flu Outbreak Predictions (Next {} weeks): {}".format(prediction_weeks, urban_predictions))
    print("Rural Flu Outbreak Predictions (Next {} weeks): {}".format(prediction_weeks, rural_predictions))

if __name__ == "__main__":
    urban_data_path = "urban_social_media_data.csv"
    rural_data_path = "rural_social_media_data.csv"
    text_column = "text"
    time_column = "date"
    flu_cases_column = "flu_cases"
    main(urban_data_path, rural_data_path, text_column, time_column, flu_cases_column)