# preprocessing.py

def scale_data(df):

    to_scale = df.drop(columns=['kfold','toxic', 'FormalCharge'])
    scaler = StandardScaler()
    scaled_values  = scaler.fit_transform(to_scale)
    scaled_df = pd.DataFrame(scaled_values, columns = to_scale.columns)
    final_df = df[['kfold', 'toxic', 'FormalCharge']].join(scaled_df)

    return scaled_df


