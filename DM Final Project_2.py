#Load libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi

# Unpickle clean dataframe
data = pd.read_pickle('data')
print(type(data))

# EDA

# TARGET
# Graph 1 - Countplot of the target
target='diagnosis'
countB = np.sum(data[target] == 'B')
countM = np.sum(data[target] == 'M')
ax = sns.countplot(x=data[target], palette=["#FF0000", "#FFC0CB"])
sns.set_style('whitegrid')
sns.despine()
total = len(data['diagnosis'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(round((p.get_height()/total)*100, 2))
        x = p.get_x() + (p.get_width()/2)-0.05
        y = p.get_y() + p.get_height() + 10
        ax.annotate(percentage, (x, y))
plt.xlabel('Diagnosis')
plt.ylabel('Total count')

#plt.title('Histogram of Diagnosis')
plt.savefig('countplot_target.png')
plt.show()

# FEATURES
# Graph 2 - Subset 'Mean', Boxplot of all features
plt.figure(figsize=(20, 20))
plt.subplot(2, 5, 1)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['radius_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 2)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['perimeter_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 3)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['area_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 4)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['texture_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 5)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['smoothness_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 6)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['compactness_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 7)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['concavity_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 8)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['concave points_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 9)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['symmetry_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 10)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['fractal_dimension_mean'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.savefig('boxplots_mean.png')
plt.show()

# Graph 3 - Subset 'Worst', Boxplot of all features
plt.figure(figsize=(15, 15))
plt.subplot(2, 5, 1)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['radius_worst'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 2)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['perimeter_worst'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 3)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['area_worst'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 4)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['texture_worst'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 5)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['smoothness_worst'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 6)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['compactness_worst'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 7)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['concavity_worst'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 8)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['concave points_worst'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 9)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['symmetry_worst'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 10)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['fractal_dimension_worst'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.savefig('boxplots_worst.png')
plt.show()

# Graph 4 - Subset 'Standard Error', Boxplot of all features
plt.figure(figsize=(15, 15))
plt.subplot(2, 5, 1)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['radius_se'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 2)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['perimeter_se'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 3)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['area_se'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 4)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['texture_se'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 5)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['smoothness_se'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 6)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['compactness_se'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 7)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['concavity_se'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 8)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['concave points_se'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 9)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['symmetry_se'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.subplot(2, 5, 10)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['fractal_dimension_se'])
sns.boxplot(x = "variable", y = "value", hue="diagnosis", data= melted_data, palette=["#FF0000", "#FFC0CB"])

plt.savefig('boxplots_SE.png')
plt.show()

# Graph 5 - Subset "Mean", Histogram of radius, perimeter, area and texture
plt.figure(figsize=(15, 15))
plt.subplot(2, 3, 1)
radius_mean_m = plt.hist(data[data["diagnosis"] == "M"].radius_mean, bins=30, fc = "#FF0000", label = "Malignant")
radius_mean_b = plt.hist(data[data["diagnosis"] == "B"].radius_mean, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean")

plt.subplot(2, 3, 2)
perimeter_mean_m = plt.hist(data[data["diagnosis"] == "M"].perimeter_mean, bins=30, fc = "#FF0000", label = "Malignant")
perimeter_mean_b = plt.hist(data[data["diagnosis"] == "B"].perimeter_mean, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Perimeter Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Perimeter Mean")

plt.subplot(2, 3, 3)
area_mean_m = plt.hist(data[data["diagnosis"] == "M"].area_mean, bins=30, fc = "#FF0000", label = "Malignant")
area_mean_b = plt.hist(data[data["diagnosis"] == "B"].area_mean, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Area Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Area Mean")

plt.subplot(2, 3, 5)
texture_mean_m = plt.hist(data[data["diagnosis"] == "M"].texture_mean, bins=30, fc = "#FF0000", label = "Malignant")
texture_mean_b = plt.hist(data[data["diagnosis"] == "B"].texture_mean, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Texture Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Texture Mean")

plt.savefig('histograms_mean.png')
plt.show()

# Graph 6 - Subset "Worst", Histogram of radius, perimeter, area and texture
plt.figure(figsize=(15, 15))
plt.subplot(2, 3, 1)
radius_worst_m = plt.hist(data[data["diagnosis"] == "M"].radius_worst, bins=30, fc = "#FF0000", label = "Malignant")
radius_worst_b = plt.hist(data[data["diagnosis"] == "B"].radius_worst, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Radius Worst Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Worst")

plt.subplot(2, 3, 2)
perimeter_worst_m = plt.hist(data[data["diagnosis"] == "M"].perimeter_worst, bins=30, fc = "#FF0000", label = "Malignant")
perimeter_worst_b = plt.hist(data[data["diagnosis"] == "B"].perimeter_worst, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Perimeter Worst Values")
plt.ylabel("Frequency")
plt.title("Histogram of Perimeter Worst")

plt.subplot(2, 3, 3)
area_worst_m = plt.hist(data[data["diagnosis"] == "M"].area_worst, bins=30, fc = "#FF0000", label = "Malignant")
area_worst_b = plt.hist(data[data["diagnosis"] == "B"].area_worst, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Area Worst Values")
plt.ylabel("Frequency")
plt.title("Histogram of Area Worst")

plt.subplot(2, 3, 5)
texture_worst_m = plt.hist(data[data["diagnosis"] == "M"].texture_worst, bins=30, fc = "#FF0000", label = "Malignant")
texture_worst_b = plt.hist(data[data["diagnosis"] == "B"].texture_worst, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Texture Worst Values")
plt.ylabel("Frequency")
plt.title("Histogram of Texture Worst")

plt.savefig('histograms_worst.png')
plt.show()

# Graph 7 - Subset "Standard Error", Histogram of radius, perimeter, area and texture
plt.figure(figsize=(15, 15))
plt.subplot(2, 3, 1)
radius_se_m = plt.hist(data[data["diagnosis"] == "M"].radius_se, bins=30, fc = "#FF0000", label = "Malignant")
radius_se_b = plt.hist(data[data["diagnosis"] == "B"].radius_se, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Radius SE Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius SE")

plt.subplot(2, 3, 2)
perimeter_se_m = plt.hist(data[data["diagnosis"] == "M"].perimeter_se, bins=30, fc = "#FF0000", label = "Malignant")
perimeter_se_b = plt.hist(data[data["diagnosis"] == "B"].perimeter_se, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Perimeter SE Values")
plt.ylabel("Frequency")
plt.title("Histogram of Perimeter SE")

plt.subplot(2, 3, 3)
area_se_m = plt.hist(data[data["diagnosis"] == "M"].area_se, bins=30, fc = "#FF0000", label = "Malignant")
area_se_b = plt.hist(data[data["diagnosis"] == "B"].area_se, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Area SE Values")
plt.ylabel("Frequency")
plt.title("Histogram of Area SE")

plt.subplot(2, 3, 5)
texture_se_m = plt.hist(data[data["diagnosis"] == "M"].texture_se, bins=30, fc = "#FF0000", label = "Malignant")
texture_se_b = plt.hist(data[data["diagnosis"] == "B"].texture_se, bins=30, fc = "#FFC0CB", label = "Benign")
plt.legend()
plt.xlabel("Texture SE Values")
plt.ylabel("Frequency")
plt.title("Histogram of Texture SE")

plt.savefig('histograms_se.png')
plt.show()

# Graph 8 - Subset 'Mean', Polar of smmothness, compactness, concavity, concave points, symmetry and fractal dimension
# Prepare data
y = data['diagnosis']
subset_mean_21 = data.iloc[:, 5:11]
subset_mean_22 = subset_mean_21.multiply(10)
subset_mean_2 = pd.concat([y, subset_mean_22], axis=1)
# Create background
categories = list(subset_mean_2)[1:]
N = len(categories)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
# Put the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
# Draw one axe per variable + add labels labels
plt.xticks(angles[:-1], categories)
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0, 1, 2, 3], ["0", "1", "2", "3"], color="grey", size=7)
plt.ylim(0, 4)
# Plot each individual = each line of the data
# Ind1
values = subset_mean_2.loc[0].drop('diagnosis').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Malignant", color="#FF0000")
ax.fill(angles, values, "#FF0000", alpha=0.1)
# Ind2
values = subset_mean_2.loc[1].drop('diagnosis').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, ls='--', label="Benign", color="#FF1493")
ax.fill(angles, values, color="#FF1493", alpha=0.1)
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
palette=["#FF0000", "#FFC0CB"]

plt.savefig('polar_mean.png')
plt.show()

# Graph 9 - Subset 'Worst', Polar of smmothness, compactness, concavity, concave points, symmetry and fractal dimension
# Prepare data
y = data['diagnosis']
subset_worst_21 = data.iloc[:, 25:31]
subset_worst_22 = subset_worst_21.multiply(10)
subset_worst_2 = pd.concat([y, subset_worst_22], axis=1)
# Create background
categories = list(subset_worst_2)[1:]
N = len(categories)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
# Put the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
# Draw one axe per variable + add labels labels
plt.xticks(angles[:-1], categories)
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ["0", "1", "2", "3", "4", "5", "6", "7"], color="grey", size=7)
plt.ylim(0, 8)
# Plot each individual = each line of the data
# Ind1
values = subset_worst_2.loc[0].drop('diagnosis').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Malignant", color="#FF0000")
ax.fill(angles, values, "#FF0000", alpha=0.1)
# Ind2
values = subset_worst_2.loc[1].drop('diagnosis').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, ls='--', label="Benign", color="#FF1493")
ax.fill(angles, values, color="#FF1493", alpha=0.1)
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.savefig('polar_worst.png')
plt.show()

# Graph 10 - Subset 'Standard Error', Polar of smmothness, compactness, concavity, concave points, symmetry and fractal dimension
# Prepare data
y = data['diagnosis']
subset_se_21 = data.iloc[:, 15:21]
subset_se_22 = subset_se_21.multiply(100)
subset_se_2 = pd.concat([y, subset_se_22], axis=1)
# Create background
categories = list(subset_se_2)[1:]
N = len(categories)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
# Put the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
# Draw one axe per variable + add labels labels
plt.xticks(angles[:-1], categories)
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ["0", "1", "2", "3", "4", "5", "6", "7"], color="grey", size=7)
plt.ylim(0, 8)
# Plot each individual = each line of the data
# Ind1
values = subset_se_2.loc[0].drop('diagnosis').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Malignant", color="#FF0000")
ax.fill(angles, values, "#FF0000", alpha=0.1)
# Ind2
values = subset_se_2.loc[1].drop('diagnosis').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, ls='--', label="Benign", color="#FF1493")
ax.fill(angles, values, color="#FF1493", alpha=0.1)
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.savefig('polar_se.png')
plt.show()

# Graph 11 - Subset Mean - Correlation Plot
subset_mean = data.iloc[:, 1:11]
f, ax = plt.subplots(figsize = (18, 18))
sns.heatmap(subset_mean.corr(), annot=True, linewidths=0.5, fmt= ".1f", ax=ax, cmap='Blues')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map Subset "Mean"', fontdict = {'fontsize' : 25})

plt.savefig('corr_mean.png')
plt.show()

# Graph 12 - Subset Worst - Correlation Plot
subset_worst = data.iloc[:, 21:31]
f, ax = plt.subplots(figsize = (18, 18))
sns.heatmap(subset_worst.corr(), annot=True, linewidths=0.5, fmt= ".1f", ax=ax, cmap='Blues')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map Subset "Worst"', fontdict = {'fontsize' : 25})

plt.savefig('corr_worst.png')
plt.show()

# Graph 13 - Subset Standard Error(SE) - Correlation Plot
subset_se = data.iloc[:, 11:21]
f, ax = plt.subplots(figsize = (18, 18))
sns.heatmap(subset_se.corr(), annot=True, linewidths=0.5, fmt= ".1f", ax=ax, cmap='Blues')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map Subset "Standard Error"', fontdict = {'fontsize' : 25})

plt.savefig('corr_se.png')
plt.show()

# Graph 14 - Correlation Plot of All Features
f, ax = plt.subplots(figsize = (18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt= ".1f", ax=ax, cmap='Blues')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map All Features', fontdict = {'fontsize' : 25})

plt.savefig('corr_all.png')
plt.show()

# Most frequent values in malignant tumors
frequent_malignant_radius_mean = radius_mean_m[0].max()
index_frequent_malignant_radius_mean = list(radius_mean_m[0]).index(frequent_malignant_radius_mean)
most_frequent_maligant_radius_mean = radius_mean_m[1][index_frequent_malignant_radius_mean]
print("Most frequent malignant radius mean is: ", most_frequent_maligant_radius_mean)

frequent_malignant_perimeter_mean = perimeter_mean_m[0].max()
index_frequent_malignant_perimeter_mean = list(perimeter_mean_m[0]).index(frequent_malignant_perimeter_mean)
most_frequent_maligant_perimeter_mean = perimeter_mean_m[1][index_frequent_malignant_perimeter_mean]
print("Most frequent malignant perimeter mean is: ", most_frequent_maligant_perimeter_mean)

frequent_malignant_area_mean = area_mean_m[0].max()
index_frequent_malignant_area_mean = list(area_mean_m[0]).index(frequent_malignant_area_mean)
most_frequent_maligant_area_mean = area_mean_m[1][index_frequent_malignant_area_mean]
print("Most frequent malignant area mean is: ", most_frequent_maligant_area_mean)

frequent_malignant_texture_mean = texture_mean_m[0].max()
index_frequent_malignant_texture_mean = list(texture_mean_m[0]).index(frequent_malignant_texture_mean)
most_frequent_maligant_texture_mean = texture_mean_m[1][index_frequent_malignant_texture_mean]
print("Most frequent malignant texture mean is: ", most_frequent_maligant_texture_mean)

frequent_malignant_radius_worst = radius_worst_m[0].max()
index_frequent_malignant_radius_worst = list(radius_worst_m[0]).index(frequent_malignant_radius_worst)
most_frequent_maligant_radius_worst = radius_worst_m[1][index_frequent_malignant_radius_worst]
print("Most frequent malignant radius worst is: ", most_frequent_maligant_radius_worst)

frequent_malignant_perimeter_worst = perimeter_worst_m[0].max()
index_frequent_malignant_perimeter_worst = list(perimeter_worst_m[0]).index(frequent_malignant_perimeter_worst)
most_frequent_maligant_perimeter_worst = perimeter_worst_m[1][index_frequent_malignant_perimeter_worst]
print("Most frequent malignant perimeter worst is: ", most_frequent_maligant_perimeter_worst)

frequent_malignant_area_worst = area_worst_m[0].max()
index_frequent_malignant_area_worst = list(area_worst_m[0]).index(frequent_malignant_area_worst)
most_frequent_maligant_area_worst = area_worst_m[1][index_frequent_malignant_area_worst]
print("Most frequent malignant area worst is: ", most_frequent_maligant_area_worst)

frequent_malignant_texture_worst = texture_worst_m[0].max()
index_frequent_malignant_texture_worst = list(texture_worst_m[0]).index(frequent_malignant_texture_worst)
most_frequent_maligant_texture_worst = texture_worst_m[1][index_frequent_malignant_texture_worst]
print("Most frequent malignant texture worst is: ", most_frequent_maligant_texture_worst)

frequent_malignant_radius_se = radius_se_m[0].max()
index_frequent_malignant_radius_se = list(radius_se_m[0]).index(frequent_malignant_radius_se)
most_frequent_maligant_radius_se = radius_se_m[1][index_frequent_malignant_radius_se]
print("Most frequent malignant radius SE is: ", most_frequent_maligant_radius_se)

frequent_malignant_perimeter_se = perimeter_se_m[0].max()
index_frequent_malignant_perimeter_se = list(perimeter_se_m[0]).index(frequent_malignant_perimeter_se)
most_frequent_maligant_perimeter_se = perimeter_se_m[1][index_frequent_malignant_perimeter_se]
print("Most frequent malignant perimeter SE is: ", most_frequent_maligant_perimeter_se)

frequent_malignant_area_se = area_se_m[0].max()
index_frequent_malignant_area_se = list(area_se_m[0]).index(frequent_malignant_area_se)
most_frequent_maligant_area_se = area_se_m[1][index_frequent_malignant_area_se]
print("Most frequent malignant area SE is: ", most_frequent_maligant_area_se)

frequent_malignant_texture_se = texture_se_m[0].max()
index_frequent_malignant_texture_se = list(texture_se_m[0]).index(frequent_malignant_texture_se)
most_frequent_maligant_texture_se = texture_se_m[1][index_frequent_malignant_texture_se]
print("Most frequent malignant texture SE is: ", most_frequent_maligant_texture_se)

