import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.tree import plot_tree
import numpy as np

dtcsv = pd.read_csv("enfinnnn.csv")
X = dtcsv.drop(labels=['category', 'ea'], axis=1)
y = dtcsv['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model = DecisionTreeClassifier(criterion="log_loss", max_depth=5, min_samples_leaf=3)
model.fit(X_train, y_train)

plot_tree(model, feature_names=
['prod1AAM', 'prod2AAM', 'react1AAM', 'react2AAM'], class_names=["1", "2", "3", "4"], filled=True)
plt.show()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score: :", f1)


print(model.predict([[35,3566167161116111,35,3566167161116111]])) # 3
print(model.predict([[9,"356616111611161116111",35,"96616111611161116111"]])) # 4
print(model.predict([[35,"666111111",9,"356611161111"]])) # 1
print(model.predict([[17,"1766171171111",17,"1766171171111"]])) # 3
print(model.predict([[17,"96616111611161116111",9,"176616111611161116111"]])) # 2
print(model.predict([[17,"1661788161116111",1,"17661788161116111"]])) # 2



# Mise en page des données
"""import pandas
import pysmiles
import csv

periodic_table = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118
}
fields = ["prod1AAM", "prod2AAM", "react1AAM", "react2AAM", "ea"]
dict_trad = []
with open('dataset2.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ea = str(round(float(row['ea'])))
        mols = [pysmiles.read_smiles(row["prod1AAM"]),pysmiles.read_smiles(row["prod2AAM"]),
        pysmiles.read_smiles(row["react1AAM"]),pysmiles.read_smiles(row["react2AAM"])]
        readed = []
        for mol in mols:
            readed.append(mol.nodes(data='element'))
        atom_trad = ""
        row_trad = []
        for _ in readed:
                for atom in _:
                    atom_trad += str(periodic_table[atom[1]])
                row_trad.append(atom_trad)
                atom_trad = ""
        row_trad.append(ea)
        dict_trad.append(row_trad)
        row_trad = []
print(dict_trad)

with open("dataset.csv" , "w") as f1:
    writer = csv.writer(f1)
    writer.writerow(fields)
    writer.writerows(dict_trad)"""

"""
import csv
all_ea = []
with open("dataset.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_ea.append(int(row['ea']))

from statistics import stdev

# sort the data, for simplicity
data = sorted(all_ea)

# create a list of the gaps between the consecutive values
gaps = [y - x for x, y in zip(data[:-1], data[1:])]
# have python calculate the standard deviation for the gaps
sd = stdev(gaps)

# create a list of lists, put the first value of the source data in the first
lists = [[data[0]]]
for x in data[1:]:
    # if the gap from the current item to the previous is more than 1 SD
    # Note: the previous item is the last item in the last list
    # Note: the '> 1' is the part you'd modify to make it stricter or more relaxed
    if (x - lists[-1][-1]) / sd > 1:
        # then start a new list
        lists.append([])
    # add the current item to the last list in the list
    lists[-1].append(x)

print(lists)
dict = []
import csv
with open('dataset.csv', "r+") as f:
    reader = csv.DictReader(f)
    for row in reader:
        el = []
        for _ in row.values():
            el.append(_)
        if int(row["ea"]) <= 15:
            el.append(1)
        elif int(row["ea"]) <= 30:
            el.append(2)
        elif int(row["ea"]) <= 45:
            el.append(3)
        elif int(row["ea"]) <= 60:
            el.append(4)
        dict.append(el)


with open("enfinnnn.csv", "w") as f2:
    writer = csv.writer(f2)
    writer.writerow(["prod1AAM", "prod2AAM", "react1AAM", "react2AAM", "ea", "category"])
    writer.writerows(dict)
"""

# Scénario d'usage
# Ici on cherche à deviner la catégorie de
# [Cl-:1],
# [C:4](=[C:5]([C:11]([H:12])([H:13])[H:14])[C:21]([H:22])([H:23])[H:24])([H:31])[N:41]([H:42])[H:43],
# [F-:2],
# [Cl:1][C:5]([C:4]([H:3])([H:31])[N:41]([H:42])[H:43])([C:11]([H:12])([H:13])[H:14])[C:21]([H:22])([H:23])[H:24],
# 8, qui est donc de catégorie 1

# Recherche des paramètres
"""
print("-"*20)
for mdepth in [1,2,3,4,5,"6",7,8,9,10,11,12,13,14]:
    model = DecisionTreeClassifier(max_depth=mdepth)
    model = model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
ev = []
for i in range(100):

    val =[]
    for criter in ["entropy", "gini", "log_loss"]:
        model = DecisionTreeClassifier(criterion=criter)
        model = model.fit(X_train, y_train)
        val.append(model.score(X_test, y_test))
    ev.append(val)

print(ev)
vals=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for it in ev:
    for i in range(len(it)):
        vals[i] = (vals[i] + it[i])/2
print(vals)
print(max(vals))"""
# KFold, n'a pas fonctionné comme voulu car les scores était plus bas que prévu
"""for i, (train_index, test_index) in enumerate(kf.split(dtcsv)):
    print(f"Fold {i}:")
    print(f" Train: index={train_index}")
    print(f" Test: index={test_index}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1.0)
    recall = recall_score(y_test, y_pred, average="weighted",zero_division=1.0)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("Accuracy :", accuracy)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1 Score: :", f1)"""


# Deux manières de le faire, la première permets de le récuperer dans une variable
# 1er méthode
#tree_rules = export_text(model,feature_names = ['prod1AAM', 'prod2AAM', 'react1AAM', 'react2AAM'])
#print(tree_rules)
# 2eme méthode
"""
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

tree_to_code(model, feature_names=['prod1AAM', 'prod2AAM', 'react1AAM', 'react2AAM'])

# Visualiser les règles : Une règle par ligne sous une forme compréhensible par l'humain
# Source : https://mljar.com/blog/extract-rules-decision-tree/
# Définition fonction get_rules

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # trier selon le nombre d'exemples
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules

rules = get_rules(model,['prod1AAM', 'prod2AAM', 'react1AAM', 'react2AAM'] ,     [1,2,3,4])
print(rules)
with open("rules.txt", "w") as fr:
  i=1
  for r in rules:
    print(r)
    fr.write("Regle{} {}\n".format(i,r))
    i = i + 1

def parse_rule(rule):
    # Extract the name of the rule
    rule_name = re.match(r'^\w+', rule).group(0)

    # Extract the condition and the action
    match = re.search(r'if\s+\((.*)\)\s+then\s+class:\s+(\w+)', rule, re.IGNORECASE)
    if not match:
        raise ValueError(f"Rule format is incorrect: {rule}")

    condition_part, action_part = match.groups()

    # Extract the class name
    class_name = action_part.strip()

    # Split multiple conditions with "AND"
    conditions = re.split(r'\s+and\s+', condition_part, flags=re.IGNORECASE)

    # Construct the CLIPS rule
    clips_rule = f'(defrule {rule_name}\n  (Data '

    # Variable set to avoid duplicates in slots
    variables = set()
    for condition in conditions:
        match = re.match(r'\(?(\w+)\s*(<=|>=|>|<|==)\s*([\d.]+)\)?', condition.strip())
        if not match:
            raise ValueError(f"Condition format is incorrect: {condition}")

        attribute, operator, value = match.groups()
        variable = f'?{attribute}'
        if attribute not in variables:
            clips_rule += f'({attribute} {variable}) '
            variables.add(attribute)

    clips_rule = clips_rule.strip() + ')\n'

    # Add the combined test
    clips_rule += '  (test (and '
    for condition in conditions:
        match = re.match(r'\(?(\w+)\s*(<=|>=|>|<|==)\s*([\d.]+)\)?', condition.strip())
        if match:
            attribute, operator, value = match.groups()
            variable = f'?{attribute}'
            clips_rule += f'({operator} {variable} {value}) '
    clips_rule = clips_rule.strip() + '))\n'

    clips_rule += f'=>\n  (assert (class {class_name})))\n'

    return clips_rule


def convert_rules_to_clips(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if line:  # Ignore les lignes vides
                clips_rule = parse_rule(line)
                outfile.write(clips_rule + '\n')


# Exemple d'utilisation
input_file = 'rules.txt'
output_file = 'rules.clp'
convert_rules_to_clips(input_file, output_file)

import pandas as pd


csv_file_path = 'enfinnnn.csv'  # Update with the path to your CSV file


# Example usage

def translate_csv_to_clips_facts_string(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Define the features and label
    features = ['prod1AAM', 'prod2AAM', 'react1AAM', 'react2AAM']
    label = 'category'

    # Initialize the CLIPS fact string
    clips_facts = "(deftemplate Data\n"

    # Define the template slots
    for feature in features:
        clips_facts += f"  (slot {feature})\n"
    clips_facts += f"  (slot {label})\n"
    clips_facts += ")\n\n"

    # Add the facts
    for index, row in df.iterrows():
        fact = "(assert (Data "
        for feature in features:
            value = row[feature]
            fact += f"({feature} {value}) "
        fact += f"({label} {row[label]})"
        fact += "))\n"
        clips_facts += fact

    # Ensure the entire string is correctly closed
    clips_facts += "\n"

    return clips_facts


clips_facts = translate_csv_to_clips_facts_string(csv_file_path)

# Print the CLIPS facts string
print(clips_facts)"""
