import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Defining Function for Model
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    
    # Creating Model DecisionTreeRegressor
    # Define model. Specify a number for random_state to ensure same results each run
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    
    # Fitting model
    model.fit(train_X, train_y)

    # Making prediction using the model
    preds_val = model.predict(val_X)

    # Validating Model
    mae = mean_absolute_error(val_y, preds_val)
    
    # Returning The Error
    return(mae)


## Importing Dataset
melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

# To see the data distribution uncomment the line below
# melbourne_data.describe()

# To see the columns in the dataset uncomment the line below
# melbourne_data.columns

# Dropping Missing Values
melbourne_data = melbourne_data.dropna(axis=0)

# Selecting Prediction Values
# Here are taking price column and putting it in "y"
y = melbourne_data.Price

# Choose the columns which you want to predict for
# To see the available columns see use the descibe method
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# To see the data distribution in selected columns uncomment the line below
# X.describe()

# To see first few values in "X" uncomment the line below
# X.head()

# Splitting Data into training and testing
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Defining Tree Length
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 5000]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

# Printing Best Tree Length
print(scores)
best_tree_size = min(scores, key=scores.get)
print("Best Tree Depth: %d" %(best_tree_size))

# Defining Final Model Using Best Tree Size
final_model = RandomForestRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)

# Perdict Using this code Below
# preds_val = model.predict(val_X)