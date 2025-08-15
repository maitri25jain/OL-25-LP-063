# Importing libraries
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

#Loading dataset and models
df=joblib.load("Models & Dataset/df.pkl")
clfc_model= joblib.load("Models & Dataset/classification_model.pkl")
reg_model= joblib.load("Models & Dataset/regression_model.pkl")

st.set_page_config(page_title= "Mental Health in Tech", layout='wide')

# Sidebar navigation
st.sidebar.title(" Menu ")
menu=st.sidebar.radio("Go to:",
    ["💡Home", "🔍Exploratory Data Analysis", '📈Regression Task', '🗃️Classification Task', '🧠Clustering Personas'])

# 💡Home
if menu== "💡Home":
    st.title("Mental Wellness Project")
    st.header("OpenLearn Capstone Project")
    st.divider()
    st.header("Project Overview")

    st.subheader("Case Study")
    st.write(""" 
            As a Machine Learning Engineer at NeuronInsights Analytics, you’ve been contracted by a coalition of
            leading tech companies including CodeLab, QuantumEdge, and SynapseWorks. Alarmed by rising burnout,
            disengagement, and attrition linked to mental health, the consortium seeks data-driven strategies to
            proactively identify and support at-risk employees. Your role is to analyze survey data from over 1,500 tech
            professionals, covering workplace policies, personal mental health history, openness to seeking help, and
            perceived employer support.
    """)

    st.subheader("Objective")
    st.markdown(""" 
            To understand the key factors influencing mental health issues among employees in the tech industry and
            build data-driven solutions for:
            - **Classification Task:** Predict whether an individual is likely to seek mental health treatment.
            - **Regression Task:** Predict the age of an individual based on personal and workplace attributes,
            supporting age-targeted intervention design.
            - **Unsupervised Task:** Segment tech employees into distinct clusters based on mental health indicators
            to aid in tailored HR policies.
    """, unsafe_allow_html=True)


    st.subheader("Dataset Overview")
    st.markdown(""" 
             **Dataset Source:** Mental Health in Tech Survey (https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)<br>
             Collected by **OSMI** (Open Sourcing Mental Illness)<br><br>
             Features include:
                - Demographic details (age, gender, country)
                - Workplace environment (mental health benefits, leave policies)
                - Personal experiences (mental illness, family history)
                - Attitudes towards mental health
    """, unsafe_allow_html=True)

# 🔍Exploratory Data Analysis
elif menu== "🔍Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.divider()
    df_raw= pd.read_csv("survey.csv")
    df_clean=df

    st.subheader("Raw Dataset")
    st.markdown(f""" 
        The original dataset, sourced from the **OSMI Mental Health in Tech Survey**, contains  
    **{df_raw.shape[0]} responses** and **{df_raw.shape[1]} features**.  
    It captures demographic details, workplace policies, and personal mental health experiences.
    """)

    st.markdown("**Missing Values in Raw Dataset:**")
    missing_vals_raw = df_raw.isnull().sum()
    missing_vals_raw = missing_vals_raw[missing_vals_raw > 0].sort_values(ascending=False)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.table(missing_vals_raw.to_frame("Missing Count").reset_index())
    
    with st.expander("View Raw Dataset Preview"):
        st.dataframe(df_raw.head())
    st.divider()

    st.subheader("Data Cleaning")
    st.markdown(f"""
    After cleaning, the dataset now has **{df_clean.shape[0]} responses** and **{df_clean.shape[1]} features**.  
    Outliers were handled, missing values were imputed or removed, and categorical variables were standardized.
    """)
    
    removed_features = ['Timestamp', 'comments', 'state', 'self_employed','tech_company'] 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Features Used:**")
        for col in df_clean.columns:
            if col not in removed_features:
                st.markdown(f"- {col}")
    with col2:
        st.markdown("**Features Removed:**")
        for col in removed_features:
            st.markdown(f"- {col}")
    
    with st.expander("📄 View Cleaned Dataset Preview"):
        st.dataframe(df_clean.head())
    
    st.divider()

    st.subheader("📈 Univariate Analysis")
    st.image("Images/univariate1.png", caption="Demographics & Treatment Distribution", use_container_width=True)
    st.image("Images/univariate2.png", caption="Company Size, Leave Policies, Awareness", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        - **Mean Age:** {round(df_clean['Age'].mean(), 2)} years  
        - **Top 5 Countries:** {', '.join(df_clean['Country'].value_counts().head(5).index)}  
        - **Treatment Seeking:** {df_clean['treatment'].value_counts(normalize=True).mul(100).round(1).to_dict()}
        """)
    with col2:
        st.markdown(f"""
        - **Gender Split:** {df_clean['Gender'].value_counts(normalize=True).mul(100).round(1).to_dict()}  
        - **Family History:** {df_clean['family_history'].value_counts(normalize=True).mul(100).round(1).to_dict()}  
        - **Most Common Work Interference:** {df_clean['work_interfere'].mode()[0]}

        """)
    st.divider()

    st.subheader("📈 Bivariate Analysis")
    st.image("Images/bivariate1.png", caption="Key Features vs Treatment", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - Younger professionals (late 20s – early 30s) seek treatment more often.  
        - Females and 'Other' gender identities show higher treatment rates than males.
        """)
    with col2:
        st.markdown("""
        - High work interference strongly correlates with treatment seeking.  
        - Employees in smaller and medium-sized companies report seeking treatment more frequently than those in larger companies.
        """)
    st.divider()
    
    st.subheader("📊 Correlation Heatmap")
    st.image("Images/multivariate.png", caption="Correlation of Workplace Mental Health Factors", use_container_width=True)

# 📈Regression Task
elif menu == "📈Regression Task":
    st.title("Predict Employee's Age")
    st.divider()

    st.markdown("""
    The regression task predicts an employee’s **age** from workplace and personal attributes. This helps identify age-based risk factors for mental health in tech workplaces.
    """)

    results_df = pd.DataFrame({
        "Model": ["Linear Regression", "Ridge Regression", "Random Forest Regression"],
        "MAE": [5.37, 5.28, 5.32],
        "RMSE": [7.29, 7.19, 7.13],
        "R² Score": [-0.008, 0.019, 0.035]
    }).sort_values(by="R² Score", ascending=False)

    st.subheader("Models Trained")
    model_desc = {
        "Linear Regression": "A simple and interpretable baseline model that fits a straight-line relationship.",
        "Ridge Regression": "A regularized linear model that reduces overfitting by penalizing large coefficients.",
        "Random Forest Regression": "An ensemble of decision trees that averages results for better robustness.",
    }
    for model in results_df["Model"]:
        st.markdown(f"### **{model}**")
        st.write(model_desc[model])
        metrics = results_df[results_df["Model"] == model].iloc[0]
        st.code(f"MAE: {metrics['MAE']:.2f}\nRMSE: {metrics['RMSE']:.2f}\nR² Score: {metrics['R² Score']:.2f}")
    st.divider()
    
    best_model = results_df.iloc[0]["Model"]
    st.success(f"✅ Best Model: **{best_model}** – Used for Predictions")

    st.write("Fill in the details below to get an estimated age.")

    input_data = {}
    display_names_reg = {
        "Gender": "Gender",
        "family_history": "Family history of mental illness?",
        "treatment": "Ever sought treatment for mental health?",
        "work_interfere": "Does mental health interfere with your work?",
        "no_employees": "Company size (number of employees)",
        "remote_work": "Work remotely at least 50% of the time?",
        "benefits": "Employer provides mental health benefits?",
        "care_options": "Aware of employer’s mental health care options?",
        "wellness_program": "Wellness programs discussed at work?",
        "seek_help": "Resources provided to seek help?",
        "anonymity": "Anonymity protected when seeking treatment?",
        "leave": "Ease of taking medical leave for mental health",
        "mental_health_consequence": "Negative consequences for discussing mental health?",
        "coworkers": "Willing to discuss mental health with coworkers?",
        "supervisor": "Willing to discuss mental health with supervisors?",
        "obs_consequence": "Observed negative consequences for others?",
        "mental_health_interview": "Would you bring up a mental health issue in a job interview?",
        "phys_health_interview": "Would you bring up a physical health issue in a job interview?",
        "mental_vs_physical": "Is mental health taken as seriously as physical health at your workplace?",
        "phys_health_consequence": "Negative consequences for discussing physical health?"

    }

    for col in df.columns:
        if col in ["Age", "age_group", "company_size"]:
            continue
        options = df[col].dropna().unique().tolist()
        label = display_names_reg.get(col, col)
        input_data[col] = st.selectbox(label, options)
        
    if st.button("Predict Age"):
        input_df = pd.DataFrame([input_data])
        pred_age = reg_model.predict(input_df)[0]
        st.success(f"Estimated Age: **{int(round(pred_age))} years**")

# 🗃️Classification Task
elif menu == "🗃️Classification Task":
    st.title("Will the Employee Seek Treatment?")
    st.divider()

    st.markdown("""
    This classification task predicts whether an employee is **likely to seek mental health treatment** based on personal and workplace attributes.
    """)
    results_clfc_df = pd.DataFrame({
        "Model": [
            "Logistic Regression", 
            "Random Forest", 
            "Support Vector Machine (SVM)", 
            "K-Nearest Neighbors (KNN)"
        ],
        "Accuracy": [0.839, 0.829, 0.818, 0.743],
        "ROC-AUC Score": [0.881, 0.883, 0.860, 0.804],
        }).sort_values(by="ROC-AUC Score", ascending=False)

    model_desc_clfc = {
        "Logistic Regression": "A simple linear classifier that predicts probabilities and works well for binary classification tasks.",
        "Random Forest": "An ensemble of decision trees that improves stability and accuracy by averaging multiple models.",
        "Support Vector Machine (SVM)": "A robust classifier that finds the optimal separating boundary between classes.",
        "K-Nearest Neighbors (KNN)": "Classifies samples based on the majority label among their nearest neighbors in the feature space."
    }

    tuned_params = {
        "Logistic Regression": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        "Random Forest": {"max_depth": 5, "max_features": "sqrt", "n_estimators": 200},
        "Support Vector Machine (SVM)": {"C": 1.0, "kernel": "rbf", "gamma": "auto"},
        "K-Nearest Neighbors (KNN)": {"n_neighbors": 7, "metric": "manhattan", "weights": "distance"}
    }

    st.subheader("Models Trained")
    for model in results_clfc_df["Model"]:
        st.markdown(f"### **{model}**")
        st.write(model_desc_clfc[model])
        st.write("Tuned Hyperparameters:")
        for param, value in tuned_params[model].items():
            st.write(f" - {param}: {value}")
        metrics = results_clfc_df[results_clfc_df["Model"] == model].iloc[0]
        st.code(f"Accuracy: {metrics['Accuracy']:.3f}\nROC-AUC Score: {metrics['ROC-AUC Score']:.3f}\n")

    st.divider()
    st.markdown("## Classification Model Comparison")
    st.dataframe(results_clfc_df.style.format({
        "Accuracy": "{:.3f}",
        "ROC-AUC Score": "{:.3f}",
    }), use_container_width=True)

    st.image("Images/roc_curve.png", caption="ROC Curve for Different Models", use_container_width=True)

    best_clf_model = results_clfc_df.iloc[0]["Model"]
    st.success(f"✅ Best Model: **{best_clf_model}** – Used for Predictions")
    st.divider()

    st.write("Answer the questions below to predict likelihood.")
    input_data = {}
    display_names_clfc = {
        "Gender": "Gender",
        "family_history": "Family history of mental illness?",
        "treatment": "Ever sought treatment for mental health?",
        "work_interfere": "Does mental health interfere with your work?",
        "no_employees": "Company size (number of employees)",
        "remote_work": "Work remotely at least 50% of the time?",
        "benefits": "Employer provides mental health benefits?",
        "care_options": "Aware of employer’s mental health care options?",
        "wellness_program": "Wellness programs discussed at work?",
        "seek_help": "Resources provided to seek help?",
        "anonymity": "Anonymity protected when seeking treatment?",
        "leave": "Ease of taking medical leave for mental health",
        "mental_health_consequence": "Negative consequences for discussing mental health?",
        "coworkers": "Willing to discuss mental health with coworkers?",
        "supervisor": "Willing to discuss mental health with supervisors?",
        "obs_consequence": "Observed negative consequences for others?",
        "mental_health_interview": "Would you bring up a mental health issue in a job interview?",
        "phys_health_interview": "Would you bring up a physical health issue in a job interview?",
        "mental_vs_physical": "Is mental health taken as seriously as physical health at your workplace?",
        "phys_health_consequence": "Negative consequences for discussing physical health?",
        "company_size": "Level of company size?"
    }

    for col in df.columns:
        if col in ["age_group", "treatment"]:
            continue
        display_label= display_names_clfc.get(col,col)
        if col == "Age":
            input_data[col] = st.number_input("Enter Age", min_value=18, max_value=100)
        else:
            options= df[col].dropna().unique().tolist()
            input_data[col] = st.selectbox(display_label, options)

    if st.button("Predict Treatment Likelihood"):
        input_df = pd.DataFrame([input_data])
        prediction = clfc_model.predict(input_df)[0]
        if prediction == 1:
            st.success("✅ Likely to seek treatment")
            st.info("Recommendation: Ensure access to mental health benefits & confidential HR policies.")
        else:
            st.error("❌ Unlikely to seek treatment")
            st.warning("Recommendation: Improve workplace awareness and communication about mental health.")

# 🧠Clustering Personas
elif menu == "🧠Clustering Personas":
    st.title("Employee Mental Wellness Personas")
    st.divider()
    st.markdown("""
        The objective of this task is to segment tech employees into **distinct mental health personas** based on their survey responses. This helps HR teams and organizations design **targeted interventions** and supportive workplace policies. 
    """)

    st.subheader("Dimensionality Reduction Techniques")
    st.write("We compared the visual separation of clusters using PCA, t-SNE, and UMAP.")
    st.image("Images/dimensions.png", caption="UMAP showed the clearest, well-separated clusters.", use_container_width=True)
    st.divider()

    st.subheader("Model Comparison (Silhouette Scores)")
    st.write("""
    - **K-Means**: 0.441
    - **Agglomerative Clustering**: 0.457 
    - **DBSCAN**: 0.121 
             
    ##### The **Highest Silhouette Score** was for **Agglomerative Clustering**
    ###### The elbow method suggested **6 clusters** as the most interpretable grouping for our dataset.")

    """)
    st.image("Images/clustering.png", caption="Clusters formed by Agglomerative Clustering")

    st.subheader("Persona Descriptions")
    persona_tabs = st.tabs(["🌱Young Aware Connecters", "🔥Mid-Career Cautious Allies", "😐High-Rish Isolates",
                             "📢Burnout Veterans", "🚫Proactive Balanced Seekers", "⚖️Steady Seniors"])
    with persona_tabs[0]:
        st.markdown(""" 
        - Mostly **younger** employees (20–30), with moderate family history 
        of mental health issues but **low current treatment uptake**. They rarely report work interference.
        
        - They’re **socially open** with coworkers and supervisors, 
        creating a positive work environment.
        - However, many are **uncertain** about 
        workplace mental health policies like leave and anonymity.
        - Their needs may 
        remain unaddressed until issues escalate, making **early identification crucial**. 
        """)
    with persona_tabs[1]:
        st.markdown("""
        - Primarily in their **30s**, showing moderate mental health needs 
        and some history.
        - They maintain **open communication** with peers and supervisors, 
        but engagement with **formal resources is limited**.
        - They may **underutilize available 
        resources**, indicating an opportunity for **proactive outreach**.
        - This segment needs 
        proactive HR nudges to convert awareness into **consistent support-seeking**.
        """)
    with persona_tabs[2]:
        st.markdown(""" 
        - Spanning **30–50**, these employees face significant mental health challenges 
        — **high family history**, frequent work interference, and clear mental health consequences. 
        - They have the **highest treatment engagement** but operate in environments with **no anonymity** 
        protections and poor policy clarity.
        - They need **urgent interventions** tailored for confidentiality, flexible leaves and mental health liaisons.        
        """)
    with persona_tabs[3]:
        st.markdown(""" 
        - Older mid-career professionals (**40–50**) with **high treatment prevalence**, 
        frequent work interference, and noticeable impacts on productivity. 
        - They benefit from **supportive colleagues and supervisors** but still face uncertainty around formal leave 
        policies and anonymity.
        - Their openness is tempered by fatigue from **long-term stress exposure**.
        - **Structured burnout prevention** programs and sustained managerial training could stabilize this group.        
        """)
    with persona_tabs[4]:
        st.markdown("""
        - Comprising mainly **20–40-year-olds**, this group engages in treatment 
        and reports minimal work interference.
        - They maintain **high trust with peers and supervisors**, 
        positioning them as **champions for mental health advocacy** within teams.
        - However, they **lack full clarity** on policies.
        - This segment could be leveraged in **wellness 
        initiatives** to normalize help-seeking behaviour.
        """)
    with persona_tabs[5]:
        st.markdown(""" 
        - Predominantly **50–60**, they report **moderate treatment** use and 
        low perceived mental health consequences.
        - Their work relationships are generally 
        **supportive**, but policy understanding remains limited.
        - Many may rely on **personal 
        coping strategies** rather than formal programs. 
        - Gentle, non-intrusive **wellness 
        check-ins** and policy clarity can prevent hidden struggles.
        """)
    
    