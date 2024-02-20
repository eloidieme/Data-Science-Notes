#### Analyze DataFrame

```python
def analyze_dataframe(df : pandas.DataFrame) -> None:
    """
    Analyze a pandas DataFrame and provide a summary of its characteristics.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame to analyze.
    
    Returns:
    None
    """
    print("DataFrame Information:")
    print("----------------------")
    display(df.info(verbose=True, show_counts=True))
    print("\n")
    
    print("DataFrame Values:")
    print("----------------------")
    display(df.head(5).T)
    print("\n")
    
    print("DataFrame Description:")
    print("----------------------")
    display(df.describe().T)
    print("\n")
    
    print("Number of Null Values:")
    print("----------------------")
    display(df.isnull().sum())
    print("\n")
    
    print("Number of Duplicated Rows:")
    print("--------------------------")
    display(df.duplicated().sum())
    print("\n")
    
    print("Number of Unique Values:")
    print("------------------------")
    display(df.nunique())
    print("\n")
    
    print("DataFrame Shape:")
    print("----------------")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
```

#### Remove duplicates from DataFrame

```python
def remove_duplicates(df : pandas.DataFrame) -> pandas.DataFrame:
    """
    Remove duplicate rows from a DataFrame and print 
    the number of duplicates found and removed.
    
    Parameters:
    - df: pandas DataFrame
    
    Returns:
    - df_no_duplicates: DataFrame with duplicates removed
    """
    
    # Identify duplicates
    duplicates = df[df.duplicated()]
    
    # Print number of duplicates found and removed
    print(f"Number of duplicates found and removed: {len(duplicates)}")
    
    # Remove duplicates
    df_no_duplicates = df.drop_duplicates()
    
    return df_no_duplicates
```

#### Plot learning curves

```python
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):  
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2) train_errors, val_errors = [], []  
for m in range(1, len(X_train)):
			model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
            val_errors.append(mean_squared_error(y_val, y_val_predict))
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
```

#### Grid search cross-validation
```python
from sklearn.model_selection import GridSearchCV
param_grid = [
		{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3]}
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```

#### Custom Transformer
```python
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):  
	def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
		self.add_bedrooms_per_room = add_bedrooms_per_room 
	def fit(self, X, y=None):
		return self # nothing else to do 
	def transform(self, X):
			rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
			population_per_household = X[:, population_ix] / X[:, households_ix]
			if self.add_bedrooms_per_room:
				bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]  
				return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
			else:  
				return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

#### Sklearn Pipeline
```python
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())
        ])
```

#### Sklearn ColumnTransformer
```python
from sklearn.compose import ColumnTransformer 

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
             ("num", num_pipeline, num_attribs),
             ("cat", OneHotEncoder(), cat_attribs),
         ])
```


