  '''
    IUpdate Scoring csv for better evaluation
    df = pd.read_csv(csv_file)

    # Copy the "model Response" column values into "Score"
    df['Score'] = df['Model Response']
    df.loc[(df['Group'] == 'Rohingya') & (df['Language'] == 'en'), 'Group'] = 'Rohingya (en)'

    # Save the modified DataFrame to a new CSV file
    df.to_csv('implizite_Analyse/2_analysis/scoring_updated.csv', index=False)
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    '''