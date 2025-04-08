import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency

class Metrics:
    def __init__(self):
        self.nome = 'Metrics'

    def _converti_a_boolean(self, df):
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if sorted(unique_vals) == [0, 1]:
                df[col] = df[col].astype(bool)
        return df
    
    def aggiungi_label(self, df_reali, df_sintetici): 
        # aggiunge etichetta 1 per dati sintetici e 0 per dati reali  
        # e merge i due dataframe
        df_reali['label'] = 1
        df_sintetici['label'] = 0
 
        return pd.concat([df_reali.copy(), df_sintetici.copy()], ignore_index=True)
    
    def separa_quant_qual(self, df_orig, force_qual_vars=None):
        # separa variabili quantitative e qualitative
        
        df = self._converti_a_boolean(df_orig.copy())
       
        non_numer = df.select_dtypes(exclude=['number']).columns.tolist()
        if force_qual_vars is None:
            force_qual_vars = []

        # Add explicitly forced qualitative variables
        force_qual_vars = list(set(force_qual_vars))  # remove duplicates
    
        all_qual_vars = list(set(non_numer + force_qual_vars))
        
        # Ensure only existing columns are selected
        all_qual_vars = [col for col in all_qual_vars if col in df.columns]
        
        quant_vars = [col for col in df.columns if col not in all_qual_vars]
        
        # Split and return
        df_quant = df[quant_vars].copy()
        df_qual = df[all_qual_vars].copy()
        
        return df_quant, df_qual

    def _descr_quant(self, df):
        summary_table = pd.DataFrame({
            'Media': df.mean(skipna=True),
            'SD': df.std(skipna=True),
            'Var': df.var(skipna=True),
            'Min': df.min(skipna=True),
            'Max': df.max(skipna=True)
        })
        return summary_table
    def _plot_dist(self, dataS, dataR):
        dfR = dataR.copy()
        dfS = dataS.copy()
        dfR['group'] = 'Reali'
        dfS['group'] = 'Sintetici'
        combined = pd.concat([dfR, dfS], ignore_index=True)

        for var in dfR.columns:
            plt.figure(figsize=(5, 3))
            sns.kdeplot(data=combined, x=var, hue='group', fill=True, common_norm=False, alpha=0.5)
            plt.title(f'Distribution: {var}')
            plt.tight_layout()
            plt.show()

    def _test_ks(self, dataR, dataS, variables):
        for var in variables:
            print(f"\n Test Kolmogorov–Smirnov per {var}:")
            stat, pval = ks_2samp(dataR[var].dropna(), dataS[var].dropna())
            print(f"Statistica = {stat:.4f}, p-value = {pval:.4g}")


    def _plot_categ(self, dataRcat, dataScat, variables):
        dfR = dataRcat.copy()
        dfS = dataScat.copy()
        dfR['group'] = 'Reale'
        dfS['group'] = 'Sintetica'
        combined = pd.concat([dfR, dfS], ignore_index=True)

        melted = combined.melt(id_vars='group', value_vars=variables, var_name='Variable', value_name='Category')

        g = sns.catplot(
            data=melted,
            kind='count',
            x='Category',
            hue='group',
            col='Variable',
            col_wrap=3,
            height=4,
            aspect=1.2,
            sharey=False
        )
        g.set_titles(col_template="{col_name}")
        g.set_xticklabels(rotation=45)
        plt.tight_layout()
        plt.show()

    def evaluate_similarity(self, df, categorical_columns):
        df = df.copy()

        # Drop constant column if it exists
        if 'device_fraud_count' in df.columns and df['device_fraud_count'].nunique() == 1:
            df = df.drop(columns='device_fraud_count')

        # Ensure label is binary int
        df['label'] = df['label'].astype(int)

        # Convert selected columns to categorical
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Drop rows with NA
        df = df.dropna()

        # Prepare X and y
        X = df.drop(columns='label')
        y = df['label']

        # Encode categoricals
        X = pd.get_dummies(X, drop_first=True)

        # Fit logistic regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        # Predictions
        prob = model.predict_proba(X)[:, 1]
        pred = (prob > 0.5).astype(int)

        # Confusion matrix
        cm = confusion_matrix(y, pred)
        print("Confusion Matrix:\n", cm)

        # ROC + AUC
        fpr, tpr, _ = roc_curve(y, prob)
        auc_val = roc_auc_score(y, prob)
        print(f"AUC: {auc_val:.4f}")

        # Plot ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}", color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Curva ROC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Accuracy
        acc = np.mean(pred == y)
        print(f"Accuracy: {acc:.4f}") 

    def _test_chi2(self, dataRcat, dataScat, variables):
        results = []
        for var in variables:
            table = pd.crosstab(index=dataRcat[var], columns='Reale').join(
                    pd.crosstab(index=dataScat[var], columns='Sintetica'),
                    how='outer').fillna(0)
            stat, pval, dof, _ = chi2_contingency(table)
            results.append({
                'Variable': var,
                'Chi2': round(stat, 4),
                'p-value': round(pval, 4),
                'dof': dof
            })
        return pd.DataFrame(results)
    
    def run(self, dataR, dataS, qual_var=None):
        if qual_var is None:
            qual_var = []

        dataR = self._converti_a_boolean(dataR)
        dataS = self._converti_a_boolean(dataS)

        dataRquant, dataRcat = self.separa_quant_qual(dataR, qual_var)
        dataSquant, dataScat = self.separa_quant_qual(dataS, qual_var)

        quant_vars = dataRquant.columns.tolist()
        cat_vars = dataRcat.columns.tolist()

        unite_conLabel = self.aggiungi_label(dataR, dataS)
        self.evaluate_similarity(unite_conLabel, cat_vars)
        
        print("Statistiche descrittive - Reali")
        print(self._descr_quant(dataRquant))
        print("\nStatistiche descrittive - Sintetici")
        print(self._descr_quant(dataSquant))

        self._plot_dist(dataSquant, dataRquant)
        self._test_ks(dataRquant, dataSquant, quant_vars)
        self._plot_categ(dataRcat, dataScat, cat_vars)

        chi2_results = self._test_chi2(dataRcat, dataScat, cat_vars)
        print("\nRisultati test Chi-quadro:")
        print(chi2_results)
