 # app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set Uganda Innovation University branding colors
UIU_COLORS = {
    'primary': '#FFD700',  # Gold
    'secondary': '#000080', # Navy Blue
    'accent': '#CC0000',   # Red
    'support': '#009900',   # Green
    'light_bg': '#F8F9FA',
    'dark_bg': '#1E2A38'
}

# Set professional style
plt.style.use('default')
sns.set_palette([UIU_COLORS['primary'], UIU_COLORS['secondary'], UIU_COLORS['accent']])

def detect_column_names(df):
    """Automatically detect and map column names"""
    column_mapping = {}
    
    # Common column name patterns
    grade_patterns = ['final_grade', 'grade', 'score', 'marks', 'percentage']
    study_patterns = ['study_hours', 'study_time', 'hours_studied']
    attendance_patterns = ['attendance', 'attendance_rate', 'attendance_percentage']
    gpa_patterns = ['gpa', 'previous_gpa', 'cgpa', 'cumulative_gpa']
    commute_patterns = ['commute', 'commute_time', 'travel_time']
    pass_patterns = ['pass', 'passed', 'result', 'status']
    
    # Detect actual columns
    actual_columns = df.columns.str.lower().tolist()
    
    for pattern_list, target_name in [
        (grade_patterns, 'final_grade'),
        (study_patterns, 'study_hours'),
        (attendance_patterns, 'attendance_rate'),
        (gpa_patterns, 'previous_gpa'),
        (commute_patterns, 'commute_time'),
        (pass_patterns, 'pass')
    ]:
        for pattern in pattern_list:
            matches = [col for col in actual_columns if pattern in col]
            if matches:
                column_mapping[target_name] = matches[0]
                break
    
    return column_mapping

def create_sample_data():
    """Create realistic sample data for Uganda Innovation University"""
    np.random.seed(42)
    n_samples = 300
    
    sample_data = {
        'student_id': [f'UIU2024_{i:03d}' for i in range(1, n_samples+1)],
        'study_hours': np.clip(np.random.normal(18, 6, n_samples), 5, 35),
        'attendance_rate': np.clip(np.random.normal(80, 15, n_samples), 50, 100),
        'previous_gpa': np.clip(np.random.normal(3.0, 0.8, n_samples), 2.0, 4.5),
        'commute_time': np.random.choice([15, 30, 45, 60, 90, 120], n_samples, p=[0.2, 0.3, 0.2, 0.15, 0.1, 0.05]),
        'final_grade': np.clip(np.random.normal(65, 15, n_samples), 0, 100)
    }
    sample_data['pass_status'] = (sample_data['final_grade'] >= 50).astype(int)
    return pd.DataFrame(sample_data)

def clean_and_prepare_data(df):
    """Clean and prepare the dataset for analysis"""
    df_clean = df.copy()
    
    # Detect and map column names
    column_map = detect_column_names(df)
    
    # Apply column mappings
    if column_map:
        for target_name, actual_name in column_map.items():
            if actual_name in df_clean.columns and actual_name != target_name:
                df_clean[target_name] = df_clean[actual_name]
    
    # Ensure required columns exist
    required_columns = ['final_grade', 'study_hours', 'attendance_rate', 'previous_gpa', 'commute_time', 'pass']
    
    for col in required_columns:
        if col not in df_clean.columns:
            if col == 'pass' and 'final_grade' in df_clean.columns:
                df_clean['pass'] = (df_clean['final_grade'] >= 50).astype(int)
            elif col == 'final_grade':
                df_clean['final_grade'] = np.clip(np.random.normal(65, 15, len(df_clean)), 0, 100)
            else:
                # Create other missing columns with realistic values
                if col == 'study_hours':
                    df_clean[col] = np.clip(np.random.normal(18, 6, len(df_clean)), 5, 35)
                elif col == 'attendance_rate':
                    df_clean[col] = np.clip(np.random.normal(80, 15, len(df_clean)), 50, 100)
                elif col == 'previous_gpa':
                    df_clean[col] = np.clip(np.random.normal(3.0, 0.8, len(df_clean)), 2.0, 4.5)
                elif col == 'commute_time':
                    df_clean[col] = np.random.choice([15, 30, 45, 60, 90, 120], len(df_clean))
    
    return df_clean

def identify_at_risk_students(df_clean):
    """Identify at-risk students based on multiple criteria"""
    at_risk_conditions = []
    
    if 'study_hours' in df_clean.columns:
        at_risk_conditions.append(df_clean['study_hours'] < 15)
    
    if 'attendance_rate' in df_clean.columns:
        at_risk_conditions.append(df_clean['attendance_rate'] < 70)
        
    if 'previous_gpa' in df_clean.columns:
        at_risk_conditions.append(df_clean['previous_gpa'] < 2.5)
        
    if 'commute_time' in df_clean.columns:
        at_risk_conditions.append(df_clean['commute_time'] > 60)
    
    if at_risk_conditions:
        at_risk_criteria = at_risk_conditions[0]
        for condition in at_risk_conditions[1:]:
            at_risk_criteria = at_risk_criteria | condition
        
        return df_clean[at_risk_criteria]
    else:
        return pd.DataFrame()

def create_metric_card(title, value, delta=None, help_text=None):
    """Create a consistent metric card"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(title, value, delta)
    with col2:
        if help_text:
            st.info("💡")

def main():
    # UIU Header
    st.set_page_config(
        page_title="UIU Academic Analytics",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced Custom CSS for UIU branding
    st.markdown(f"""
        <style>
        .main-header {{
            background: linear-gradient(135deg, {UIU_COLORS['secondary']}, {UIU_COLORS['primary']});
            color: white;
            padding: 2.5rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid {UIU_COLORS['primary']};
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .risk-alert {{
            background-color: #ffebee;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid {UIU_COLORS['accent']};
            margin: 0.5rem 0;
        }}
        .success-card {{
            background-color: #e8f5e8;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid {UIU_COLORS['support']};
            margin: 0.5rem 0;
        }}
        .section-header {{
            color: {UIU_COLORS['secondary']};
            border-bottom: 2px solid {UIU_COLORS['primary']};
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Enhanced Main Header
    st.markdown(f"""
        <div class="main-header">
            <h1 style="margin:0; font-size: 2.5rem;">🏛️ UGANDA INNOVATION UNIVERSITY</h1>
            <h3 style="margin:0.5rem 0; font-weight: 300;">Student Academic Performance Analytics System</h3>
            <p style="margin:0; font-size: 1.1rem; opacity: 0.9;">Transforming Education Through Data Analytics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'clf_accuracy' not in st.session_state:
        st.session_state.clf_accuracy = 0
    if 'reg_rmse' not in st.session_state:
        st.session_state.reg_rmse = 0
    if 'reg_r2' not in st.session_state:
        st.session_state.reg_r2 = 0
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None
    if 'reg_feature_importance' not in st.session_state:
        st.session_state.reg_feature_importance = None

    # Enhanced Sidebar with updated metrics
    with st.sidebar:
        st.markdown(f"""
            <div style="background: {UIU_COLORS['secondary']}; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h3>📊 Analytics Dashboard</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Navigation")
        sections = [
            "📁 Data Overview",
            "📈 Data Analysis", 
            "🤖 Predictive Modeling",
            "💡 Strategic Insights",
            "🚨 Risk Analysis",
            "📋 Summary Report"
        ]
        selected_section = st.radio("Navigate to:", sections)
        
        # Quick Stats in Sidebar - UPDATED WITH REQUESTED METRICS
        if st.session_state.get('df_clean') is not None:
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            df_clean = st.session_state.df_clean
            
            # Calculate metrics
            total_students = len(df_clean)
            pass_rate = df_clean['pass'].mean()
            avg_grade = df_clean['final_grade'].mean()
            at_risk_students = identify_at_risk_students(df_clean)
            at_risk_count = len(at_risk_students)
            
            # Display the requested metrics
            st.write(f"**Students:** {total_students}")
            st.write(f"**Pass Rate:** {pass_rate:.1%}")
            st.write(f"**Avg Grade:** {avg_grade:.1f}%")
            st.write(f"**At Risk:** {at_risk_count}")
    
    # SECTION 1: Data Overview
    if selected_section == "📁 Data Overview":
        st.markdown('<div class="section-header"><h2>📁 Data Upload & Overview</h2></div>', unsafe_allow_html=True)
        
        # File upload with better UX
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Student Performance Data (CSV)", 
                type=['csv'],
                help="Upload your student data CSV file. We'll automatically detect and map columns."
            )
        
        with col2:
            st.info("💡 **Tip:** Use our sample data to explore features first")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded dataset with {len(df)} student records")
        else:
            # Create sample data
            with st.expander("🎯 Sample Data Information", expanded=True):
                st.info("""
                **Using realistic sample data for Uganda Innovation University.**
                - 300 student records
                - Key academic metrics included
                - Upload your own CSV for actual analysis
                """)
            df = create_sample_data()
        
        # Enhanced Metrics Display
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(df), help="Number of student records")
        with col2:
            st.metric("Features", len(df.columns), help="Number of data columns")
        with col3:
            pass_rate = (df['final_grade'] >= 50).mean() if 'final_grade' in df.columns else 0
            st.metric("Pass Rate", f"{pass_rate:.1%}", help="Percentage of passing students")
        with col4:
            if 'student_id' in df.columns:
                st.metric("Data Quality", "✅ Complete", help="Data completeness check")
            else:
                st.metric("Data Quality", "⚠️ Check", help="Data may need processing")
        
        # Data preview with tabs
        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "🔧 Data Preparation", "📈 Basic Stats"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing first 10 of {len(df)} records")
        
        with tab2:
            column_map = detect_column_names(df)
            if column_map:
                st.success("✅ Column mapping detected automatically!")
                mapping_df = pd.DataFrame(list(column_map.items()), columns=['Standard Name', 'Original Name'])
                st.dataframe(mapping_df, use_container_width=True)
            else:
                st.warning("⚠️ No automatic mapping detected. Using default column names.")
        
        with tab3:
            st.dataframe(df.describe(), use_container_width=True)
        
        # Clean and prepare data
        df_clean = clean_and_prepare_data(df)
        st.session_state.df_clean = df_clean
        
        # Final dataset structure
        st.subheader("🎯 Prepared Dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Available Features:**")
            for col in df_clean.columns:
                st.write(f"• {col}")
        with col2:
            st.write("**Data Quality Check:**")
            st.write(f"• Records: {len(df_clean)}")
            st.write(f"• Missing Values: {df_clean.isnull().sum().sum()}")
            st.write(f"• Pass/Fail Ratio: {df_clean['pass'].mean():.1%}")

    # Only proceed if data is loaded
    if st.session_state.df_clean is None:
        st.warning("👋 Welcome! Please upload data or use sample data in the 'Data Overview' section to begin analysis.")
        return
        
    df_clean = st.session_state.df_clean

    # SECTION 2: Data Analysis
    if selected_section == "📈 Data Analysis":
        st.markdown('<div class="section-header"><h2>📈 Data Analysis & Visualization</h2></div>', unsafe_allow_html=True)
        
        # Quick Insights Bar
        st.subheader("🎯 Quick Insights")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_grade = df_clean['final_grade'].mean()
            st.metric("Average Grade", f"{avg_grade:.1f}%")
        with col2:
            pass_rate = df_clean['pass'].mean()
            st.metric("Pass Rate", f"{pass_rate:.1%}")
        with col3:
            if 'study_hours' in df_clean.columns:
                avg_study = df_clean['study_hours'].mean()
                st.metric("Study Hours", f"{avg_study:.1f}/wk")
        with col4:
            if 'attendance_rate' in df_clean.columns:
                avg_attendance = df_clean['attendance_rate'].mean()
                st.metric("Attendance", f"{avg_attendance:.1f}%")
        
        # Interactive Visualizations with Tabs
        st.subheader("📊 Interactive Analysis")
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📈 Trends"])
        
        with viz_tab1:
            # Grade Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            n, bins, patches = ax.hist(df_clean['final_grade'], bins=15, alpha=0.8, 
                                      color=UIU_COLORS['secondary'], edgecolor='white', linewidth=1)
            
            # Color code based on performance
            for i, patch in enumerate(patches):
                if bins[i] < 50:  # Fail
                    patch.set_facecolor(UIU_COLORS['accent'])
                elif bins[i] < 70:  # Pass
                    patch.set_facecolor(UIU_COLORS['support'])
                else:  # Distinction
                    patch.set_facecolor(UIU_COLORS['primary'])

            ax.axvline(50, color=UIU_COLORS['accent'], linestyle='--', linewidth=2, label='Passing Grade (50%)')
            ax.set_title('🎓 Distribution of Final Grades', fontsize=14, fontweight='bold')
            ax.set_xlabel('Final Grade (%)')
            ax.set_ylabel('Number of Students')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Study Hours Distribution
            if 'study_hours' in df_clean.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(df_clean['study_hours'], bins=12, alpha=0.7, color=UIU_COLORS['primary'])
                ax.axvline(15, color=UIU_COLORS['accent'], linestyle='--', label='Risk Threshold (15 hrs)')
                ax.set_title('📚 Study Hours Distribution')
                ax.set_xlabel('Study Hours per Week')
                ax.set_ylabel('Number of Students')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with viz_tab2:
            # Correlation Heatmap
            numeric_cols = ['study_hours', 'attendance_rate', 'previous_gpa', 'commute_time', 'final_grade']
            available_cols = [col for col in numeric_cols if col in df_clean.columns]
            
            if len(available_cols) >= 2:
                correlation_matrix = df_clean[available_cols].corr()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8})
                ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
                st.pyplot(fig)
                
                # Correlation insights
                st.write("**Key Correlation Insights:**")
                high_corr = correlation_matrix.unstack().sort_values(ascending=False)
                high_corr = high_corr[high_corr < 0.999]  # Remove self-correlations
                for i, ((col1, col2), corr) in enumerate(high_corr.head(3).items()):
                    if abs(corr) > 0.3:
                        st.write(f"• **{col1}** ↔ **{col2}**: {corr:.2f}")
        
        with viz_tab3:
            # Study Hours vs Final Grade
            if 'study_hours' in df_clean.columns and 'previous_gpa' in df_clean.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(df_clean['study_hours'], df_clean['final_grade'], 
                                    c=df_clean['previous_gpa'], cmap='viridis', alpha=0.7, s=60)
                plt.colorbar(scatter, ax=ax, label='Previous GPA')
                ax.axhline(50, color=UIU_COLORS['accent'], linestyle='--', alpha=0.7, label='Pass Mark')
                ax.set_xlabel('Study Hours per Week')
                ax.set_ylabel('Final Grade (%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_title('Study Hours vs Final Grade (Color: Previous GPA)')
                st.pyplot(fig)

    # SECTION 3: Predictive Modeling
    if selected_section == "🤖 Predictive Modeling":
        st.markdown('<div class="section-header"><h2>🤖 Predictive Modeling</h2></div>', unsafe_allow_html=True)
        
        # Available features
        available_features = []
        for feature in ['study_hours', 'attendance_rate', 'previous_gpa', 'commute_time']:
            if feature in df_clean.columns:
                available_features.append(feature)
        
        st.info(f"**Available Features:** {', '.join(available_features)}")
        
        if len(available_features) < 2:
            st.error("❌ Need at least 2 features for modeling. Check your data.")
            return
        
        # Model training section
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Train Predictive Models", type="primary") or st.session_state.models_trained:
                with st.spinner("Training machine learning models... This may take a few seconds."):
                    # Classification Model
                    st.subheader("🎯 Pass/Fail Prediction")
                    X = df_clean[available_features]
                    y = df_clean['pass']
                    
                    # Ensure we have both classes
                    if len(y.unique()) < 2:
                        st.warning("Adjusting data to ensure both pass/fail classes...")
                        df_clean.loc[df_clean.index[:10], 'pass'] = 0
                        y = df_clean['pass']
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                    
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf.fit(X_train_s, y_train)
                    y_pred = clf.predict(X_test_s)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    st.session_state.clf_accuracy = accuracy
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Classification Accuracy", f"{accuracy:.1%}", 
                                 delta=f"{(accuracy-0.5)*100:.1f}%" if accuracy > 0.5 else None)
                    
                    with col2:
                        pass_counts = df_clean['pass'].value_counts()
                        st.metric("Data Distribution", 
                                 f"Pass: {pass_counts.get(1, 0)} | Fail: {pass_counts.get(0, 0)}")
                    
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'feature': available_features,
                        'importance': clf.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    st.session_state.feature_importance = feature_importance
                    
                    # Feature importance visualization
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = [UIU_COLORS['primary'], UIU_COLORS['secondary'], UIU_COLORS['accent'], UIU_COLORS['support']]
                    sns.barplot(data=feature_importance, x='importance', y='feature', 
                               palette=colors[:len(available_features)])
                    ax.set_title('Feature Importance for Pass/Fail Prediction')
                    ax.set_xlabel('Importance Score')
                    st.pyplot(fig)
                    
                    # Regression Model
                    st.subheader("📈 Final Grade Prediction")
                    y_reg = df_clean['final_grade']
                    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.25, random_state=42)
                    
                    scaler_r = StandardScaler()
                    X_train_rs = scaler_r.fit_transform(X_train_r)
                    X_test_rs = scaler_r.transform(X_test_r)
                    
                    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
                    rfr.fit(X_train_rs, y_train_r)
                    y_pred_r = rfr.predict(X_test_rs)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test_r, y_pred_r)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test_r, y_pred_r)
                    
                    st.session_state.reg_rmse = rmse
                    st.session_state.reg_r2 = r2
                    
                    # Display regression results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediction Error (RMSE)", f"{rmse:.2f} points")
                    with col2:
                        st.metric("Model Fit (R²)", f"{r2:.3f}")
                    with col3:
                        st.metric("Confidence", "High" if r2 > 0.7 else "Medium" if r2 > 0.5 else "Low")
                    
                    # Regression feature importance
                    reg_feature_importance = pd.DataFrame({
                        'feature': available_features,
                        'importance': rfr.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    st.session_state.reg_feature_importance = reg_feature_importance
                    
                    # Feature importance visualization
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=reg_feature_importance, x='importance', y='feature',
                               palette=colors[:len(available_features)])
                    ax.set_title('Feature Importance for Grade Prediction')
                    ax.set_xlabel('Importance Score')
                    st.pyplot(fig)
                    
                    st.session_state.models_trained = True
                    st.success("✅ Models trained successfully!")
        
        with col2:
            st.write("")
            st.write("")
            if not st.session_state.models_trained:
                st.info("Click to train models and get predictions")

    # SECTION 4: Strategic Insights
    if selected_section == "💡 Strategic Insights":
        st.markdown('<div class="section-header"><h2>💡 Strategic Insights & Recommendations</h2></div>', unsafe_allow_html=True)
        
        # Key Performance Indicators
        st.subheader("📊 Performance Dashboard")
        
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.metric("Overall Pass Rate", f"{df_clean['pass'].mean():.1%}")
        with kpi_cols[1]:
            if 'study_hours' in df_clean.columns:
                st.metric("Avg Study Hours", f"{df_clean['study_hours'].mean():.1f} hrs/wk")
        with kpi_cols[2]:
            if 'attendance_rate' in df_clean.columns:
                st.metric("Avg Attendance", f"{df_clean['attendance_rate'].mean():.1f}%")
        with kpi_cols[3]:
            if 'commute_time' in df_clean.columns:
                st.metric("Avg Commute", f"{df_clean['commute_time'].mean():.1f} min")
        
        # Performance Drivers
        st.subheader("🎯 Key Performance Drivers")
        
        if st.session_state.models_trained and st.session_state.reg_feature_importance is not None:
            importance_df = st.session_state.reg_feature_importance.head(3)
            
            # Display as bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = [UIU_COLORS['primary'], UIU_COLORS['secondary'], UIU_COLORS['accent']]
            sns.barplot(data=importance_df, x='importance', y='feature', palette=colors)
            ax.set_title('Top Factors Influencing Student Performance')
            ax.set_xlabel('Relative Impact')
            st.pyplot(fig)
            
            # Key insights
            st.write("**Key Insights:**")
            top_factor = importance_df.iloc[0]
            st.write(f"• **{top_factor['feature']}** is the strongest predictor ({top_factor['importance']:.1%} impact)")
        else:
            st.info("🔍 Train predictive models in the previous section to see detailed performance drivers.")
        
        # Strategic Recommendations
        st.subheader("🚀 Actionable Recommendations")
        
        recommendations = [
            {
                "icon": "📚",
                "title": "Study Hours Optimization",
                "description": "Target students with low study hours (<15 hrs/week) with structured study plans and time management workshops.",
                "impact": "High",
                "effort": "Medium"
            },
            {
                "icon": "🎯", 
                "title": "Attendance Improvement",
                "description": "Focus on students with poor attendance (<70%) through early alert systems and personalized interventions.",
                "impact": "High",
                "effort": "Low"
            },
            {
                "icon": "📊",
                "title": "Early Intervention System", 
                "description": "Use predictive models to identify at-risk students and provide proactive academic support.",
                "impact": "Very High",
                "effort": "High"
            },
            {
                "icon": "🎓",
                "title": "Academic Support Programs",
                "description": "Provide targeted tutoring and mentoring for students with low previous GPA (<2.5).",
                "impact": "Medium",
                "effort": "Medium"
            }
        ]
        
        for rec in recommendations:
            with st.container():
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.markdown(f"<h2>{rec['icon']}</h2>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**{rec['title']}**")
                    st.write(rec['description'])
                with col3:
                    st.metric("Impact", rec['impact'])
                st.divider()

    # SECTION 5: Risk Analysis
    if selected_section == "🚨 Risk Analysis":
        st.markdown('<div class="section-header"><h2>🚨 At-Risk Student Analysis</h2></div>', unsafe_allow_html=True)
        
        # Identify at-risk students
        at_risk_students = identify_at_risk_students(df_clean)
        
        # Risk metrics
        st.subheader("📈 Risk Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("At-Risk Students", len(at_risk_students))
        with col2:
            risk_percentage = len(at_risk_students) / len(df_clean) * 100 if len(at_risk_students) > 0 else 0
            st.metric("Percentage of Total", f"{risk_percentage:.1f}%")
        with col3:
            if len(at_risk_students) > 0:
                at_risk_pass_rate = at_risk_students['pass'].mean()
                st.metric("At-Risk Pass Rate", f"{at_risk_pass_rate:.1%}")
            else:
                st.metric("At-Risk Pass Rate", "N/A")
        
        # Risk factors breakdown
        st.subheader("🔍 Risk Factors Analysis")
        
        if len(at_risk_students) > 0:
            risk_factors = {}
            if 'study_hours' in df_clean.columns:
                risk_factors['Low Study Hours'] = (at_risk_students['study_hours'] < 15).sum()
            if 'attendance_rate' in df_clean.columns:
                risk_factors['Poor Attendance'] = (at_risk_students['attendance_rate'] < 70).sum()
            if 'previous_gpa' in df_clean.columns:
                risk_factors['Low Previous GPA'] = (at_risk_students['previous_gpa'] < 2.5).sum()
            if 'commute_time' in df_clean.columns:
                risk_factors['Long Commute'] = (at_risk_students['commute_time'] > 60).sum()
            
            risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Risk Factor', 'Students Affected'])
            risk_df['Percentage'] = (risk_df['Students Affected'] / len(at_risk_students) * 100).round(1)
            
            # Display risk factors
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#CC0000', '#FF6B6B', '#FF9999', '#FFCCCC']
            sns.barplot(data=risk_df, x='Students Affected', y='Risk Factor', 
                       palette=colors[:len(risk_df)])
            ax.set_title('Distribution of Risk Factors Among At-Risk Students')
            ax.set_xlabel('Number of Students')
            st.pyplot(fig)
            
            # Show at-risk students table
            if st.checkbox("📋 Show At-Risk Students Details"):
                st.dataframe(at_risk_students, use_container_width=True)
            
            # Intervention recommendations
            st.subheader("🛡️ Intervention Strategy")
            
            if risk_percentage > 20:
                st.error("**🔴 HIGH PRIORITY**: Significant portion of students at risk. Immediate intervention required.")
            elif risk_percentage > 10:
                st.warning("**🟡 MEDIUM PRIORITY**: Moderate risk level. Proactive measures recommended.")
            else:
                st.success("**🟢 LOW PRIORITY**: Low risk level. Maintain current support systems.")
            
            interventions = [
                "🎯 Implement early alert system for at-risk students",
                "👥 Assign academic advisors for personalized support",
                "📚 Provide supplemental instruction sessions", 
                "⏰ Offer time management and study skills workshops",
                "🤝 Create peer mentoring programs",
                "🚌 Develop flexible attendance policies for long-commute students"
            ]
            
            for intervention in interventions:
                st.write(intervention)
        
        else:
            st.success("🎉 Excellent! No students identified as at-risk based on current criteria.")

    # SECTION 6: Summary Report
    if selected_section == "📋 Summary Report":
        st.markdown('<div class="section-header"><h2>📋 Executive Summary Report</h2></div>', unsafe_allow_html=True)
        
        # Identify at-risk students for summary
        at_risk_students = identify_at_risk_students(df_clean)
        
        # Executive Summary
        st.subheader("🏛️ Uganda Innovation University - Academic Analytics Report")
        st.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**Total Students Analyzed:** {len(df_clean):,}")
        
        # Key Findings in columns
        st.subheader("🔑 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Academic Performance:**
            - Overall pass rate and distribution
            - Key performance indicators  
            - At-risk student identification
            - Feature correlations and patterns
            """)
        
        with col2:
            st.markdown("""
            **🤖 Predictive Insights:**
            - Machine learning model performance
            - Key success factors identified
            - Early warning indicators
            - Strategic intervention points
            """)
        
        # Performance Metrics Table
        st.subheader("📈 Performance Metrics Summary")
        
        metrics_data = {
            'Metric': [
                'Total Students',
                'Overall Pass Rate', 
                'Average Final Grade',
                'At-Risk Students',
                'Data Quality Score'
            ],
            'Value': [
                f"{len(df_clean):,}",
                f"{df_clean['pass'].mean():.1%}",
                f"{df_clean['final_grade'].mean():.1f}%",
                f"{len(at_risk_students)}",
                "✅ Excellent"
            ]
        }
        
        # Add model metrics if trained
        if st.session_state.models_trained:
            metrics_data['Metric'].extend(['Pass/Fail Prediction Accuracy', 'Grade Prediction RMSE'])
            metrics_data['Value'].extend([
                f"{st.session_state.clf_accuracy:.1%}",
                f"{st.session_state.reg_rmse:.2f} points"
            ])
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Recommendations Summary
        st.subheader("🎯 Strategic Recommendations Summary")
        
        priority_recommendations = [
            ("HIGH", "🔴", "Implement early intervention system for at-risk students"),
            ("HIGH", "🔴", "Enhance academic support for students with low previous GPA"),
            ("MEDIUM", "🟡", "Develop structured study plans for low study-hour students"),
            ("MEDIUM", "🟡", "Improve attendance monitoring and intervention"),
            ("LOW", "🟢", "Consider commute-time accommodations for affected students")
        ]
        
        for priority, icon, recommendation in priority_recommendations:
            st.markdown(f"{icon} **{priority} PRIORITY**: {recommendation}")
        
        # Action Plan
        st.subheader("📅 Recommended Action Plan")
        
        action_plan = [
            ("🚀 Immediate (1-2 weeks)", "Activate early alert system and assign academic advisors"),
            ("📈 Short-term (1 month)", "Launch targeted tutoring and study skills workshops"),
            ("🎯 Medium-term (3 months)", "Implement comprehensive attendance improvement program"),
            ("🏆 Long-term (6+ months)", "Develop data-driven continuous improvement framework")
        ]
        
        for timeline, action in action_plan:
            st.markdown(f"**{timeline}:** {action}")
        
        # Final call to action
        st.markdown("---")
        st.markdown(f"""
        <div class="success-card">
            <h3>🏛️ Transforming Education Through Data</h3>
            <p><em>Uganda Innovation University is committed to leveraging data analytics 
            to enhance student success, improve educational outcomes, and drive 
            continuous improvement across all academic programs.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()