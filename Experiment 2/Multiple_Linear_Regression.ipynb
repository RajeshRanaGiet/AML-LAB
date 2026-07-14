{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN7cMWWdWAX5lQ6JomwKtDJ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/RajeshRanaGiet/AML-LAB/blob/main/Experiment%202/Multiple_Linear_Regression.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "print(\"Please enter comma-separated numerical values for the following fields:\")\n",
        "study_hours_input = input(\"Enter the study hours (e.g., 2,4,6,8): \")\n",
        "attendance_input = input(\"Enter the attendance (e.g., 70,80,85,90): \")\n",
        "assignment_input = input(\"Enter the assignment scores (e.g., 12,15,18,20): \")\n",
        "exam_marks_input = input(\"Enter the exam marks (e.g., 45,60,75,90): \")\n",
        "\n",
        "study_hours = np.array([float(x) for x in study_hours_input.split(\",\") if x.strip()])\n",
        "attendance = np.array([float(x) for x in attendance_input.split(\",\") if x.strip()])\n",
        "assignment = np.array([float(x) for x in assignment_input.split(\",\") if x.strip()])\n",
        "exam_marks = np.array([float(x) for x in exam_marks_input.split(\",\") if x.strip()])\n",
        "\n",
        "if not (len(study_hours) == len(attendance) == len(assignment) == len(exam_marks)):\n",
        "    print(\"\\nError: All input fields must contain the exact same number of data points.\")\n",
        "elif len(study_hours) < 2:\n",
        "    print(\"\\nError: Please enter at least 2 or more data points.\")\n",
        "else:\n",
        "    data = pd.DataFrame({\n",
        "        'x1': study_hours,\n",
        "        'x2': attendance,\n",
        "        'x3': assignment\n",
        "    })\n",
        "\n",
        "    model = LinearRegression()\n",
        "    model.fit(data, exam_marks)\n",
        "\n",
        "    intercept = model.intercept_\n",
        "    coefficients = model.coef_\n",
        "\n",
        "    equation_terms = [f\"({coef:.4f} * x{i+1})\" for i, coef in enumerate(coefficients)]\n",
        "    equation_str = f\"y = {intercept:.4f} + \" + \" + \".join(equation_terms)\n",
        "\n",
        "    print(\"\\n==============================\")\n",
        "    print(\"Final Regression Equation:\")\n",
        "    print(equation_str)\n",
        "    print(\"==============================\")\n",
        "    print(f\"Intercept (b0): {intercept:.4f}\")\n",
        "\n",
        "    for i, coef in enumerate(coefficients):\n",
        "        print(f\"Slope (b{i+1}): {coef:.4f}\")\n",
        "    print()\n",
        "\n",
        "    while True:\n",
        "        choice = input(\"Do you want to enter sample data (study hours, attendance, assignment) for predicting the mark? (yes/no): \").strip().lower()\n",
        "\n",
        "        if choice in ['yes', 'y']:\n",
        "            try:\n",
        "                print(\"\\n--- Enter Values for Prediction ---\")\n",
        "                x1_val = float(input(\"Enter Study Hours (x1): \"))\n",
        "                x2_val = float(input(\"Enter Attendance (x2): \"))\n",
        "                x3_val = float(input(\"Enter Assignment Score (x3): \"))\n",
        "\n",
        "                sample_df = pd.DataFrame([{\n",
        "                    'x1': x1_val,\n",
        "                    'x2': x2_val,\n",
        "                    'x3': x3_val\n",
        "                }])\n",
        "\n",
        "                predicted_mark = model.predict(sample_df)[0]\n",
        "                print(f\">> Predicted Exam Mark (y): {predicted_mark:.4f}\\n\")\n",
        "            except ValueError:\n",
        "                print(\"Invalid input! Please enter numeric values.\\n\")\n",
        "\n",
        "        elif choice in ['no', 'n']:\n",
        "            print(\"Exiting program. Goodbye!\")\n",
        "            break\n",
        "        else:\n",
        "            print(\"Please answer with 'yes' or 'no'.\\n\")\n",
        "\n",
        "#Testing data :- 6,85,18,75"
      ],
      "metadata": {
        "id": "n6LVTPgU22xc",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "46707ed4-1268-41be-b5d8-839bd810f0cb"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Please enter comma-separated numerical values for the following fields:\n",
            "Enter the study hours (e.g., 2,4,6,8): 2,4,6,8\n",
            "Enter the attendance (e.g., 70,80,85,90): 70,80,85,90\n",
            "Enter the assignment scores (e.g., 12,15,18,20): 12,15,18,20\n",
            "Enter the exam marks (e.g., 45,60,75,90): 45,60,75,90\n",
            "\n",
            "==============================\n",
            "Final Regression Equation:\n",
            "y = 30.0000 + (7.5000 * x1) + (-0.0000 * x2) + (0.0000 * x3)\n",
            "==============================\n",
            "Intercept (b0): 30.0000\n",
            "Slope (b1): 7.5000\n",
            "Slope (b2): -0.0000\n",
            "Slope (b3): 0.0000\n",
            "\n",
            "Do you want to enter sample data (study hours, attendance, assignment) for predicting the mark? (yes/no): y\n",
            "\n",
            "--- Enter Values for Prediction ---\n",
            "Enter Study Hours (x1): 6\n",
            "Enter Attendance (x2): 85\n",
            "Enter Assignment Score (x3): 18\n",
            ">> Predicted Exam Mark (y): 75.0000\n",
            "\n",
            "Do you want to enter sample data (study hours, attendance, assignment) for predicting the mark? (yes/no): n\n",
            "Exiting program. Goodbye!\n"
          ]
        }
      ]
    }
  ]
}