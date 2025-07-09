import streamlit as st
import pandas as pd
import bcrypt
import os
import joblib
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer

# --------------------- 用户认证模块 ---------------------
USERS_FILE = "users.csv"
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password_hash", "email"]).to_csv(USERS_FILE, index=False)

def load_users():
    return pd.read_csv(USERS_FILE)

def save_user(username, password, email):
    users = load_users()
    if username in users['username'].values:
        return False
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    new_user = pd.DataFrame([[username, password_hash.decode(), email]],
                          columns=["username", "password_hash", "email"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USERS_FILE, index=False)
    return True

def verify_user(username, password):
    users = load_users()
    user = users[users['username'] == username]
    if user.empty:
        return False
    return bcrypt.checkpw(password.encode(), user.iloc[0]['password_hash'].encode())

def reset_password_by_email(email, new_password):
    users = load_users()
    user = users[users['email'] == email]
    if not user.empty:
        password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        users.loc[users['email'] == email, 'password_hash'] = password_hash
        users.to_csv(USERS_FILE, index=False)
        return True
    return False

# --------------------- 全局状态 ---------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# --------------------- 样式配置 ---------------------
def apply_global_styles():
    """简洁现代的样式方案"""
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <style>
        /* 主背景色 */
        .stApp {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* 标题样式 */
        .global-header h1 {
            color: #1e3d59;
            margin-bottom: 0.5rem;
            font-size: 2.8rem !important;
            text-align: center;
        }
        
        .global-header p {
            color: #4a6572;
            font-size: 1.5rem !important;
            margin-top: 0;
            text-align: center;
        }
        
        /* 卡片样式 */
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            border-left: 4px solid #3f87a6;
        }
        
        /* 输入框样式 */
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            padding: 12px 16px !important;
            font-size: 16px !important;
            border-radius: 8px !important;
        }
        
        /* 按钮样式 */
        .stButton button {
            background-color: #3f87a6 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton button:hover {
            background-color: #2c6a8a !important;
        }
        
        /* 页脚样式 */
        footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #eaeaea;
            color: #6c757d;
            font-size: 0.9rem;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

def render_global_header():
    """渲染全局头部组件"""
    st.markdown("""
    <div class="global-header">
        <h1>阻燃聚合物复合材料智能设计平台</h1>
        <p>Flame Retardant Composites AI Platform</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------- 首页内容 ---------------------
def show_homepage():
    apply_global_styles()
    render_global_header()
    
    # 主内容容器
    st.markdown("""
    <div style="max-width:1400px; margin:0 auto; padding:2rem;">
    """, unsafe_allow_html=True)

    # 平台简介
    st.markdown("""
    <div style="font-size:1.2rem; line-height:1.6; margin-bottom:2.5rem; text-align: center;">
        🚀 本平台融合AI与材料科学技术，致力于高分子复合材料的智能化设计，
        重点关注阻燃性能、力学性能和热稳定性的多目标优化与调控。
    </div>
    """, unsafe_allow_html=True)

    # 核心功能
    st.markdown("""
    <h2 style="font-size:1.8rem; color:#1e3d59; border-bottom: 2px solid #3f87a6; padding-bottom:0.5rem; margin-bottom:1.5rem;">
        🌟 核心功能
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="font-size:1.4rem; color:#1e3d59; margin:0 0 1rem 0;">
                🔥 智能性能预测
            </h3>
            <p style="font-size:1.1rem;">
                • 支持LOI（极限氧指数）预测<br>
                • TS（拉伸强度）预测<br>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="font-size:1.4rem; color:#1e3d59; margin:0 0 1rem 0;">
                ⚗️ 配方优化系统
            </h3>
            <p style="font-size:1.1rem;">
                • 根据输入目标推荐配方<br>
                • 支持选择配方种类<br>
                • 添加剂比例智能推荐
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 研究成果
    st.markdown("""
    <h2 style="font-size:1.8rem; color:#1e3d59; border-bottom: 2px solid #3f87a6; padding-bottom:0.5rem; margin-bottom:1.5rem;">
        🏆 研究成果
    </h2>
    <div class="feature-card">
        <p style="font-size:1.1rem;">
            Ma Weibin, Li Ling, Zhang Yu, et al.<br>
            <em>Active learning-based generative design of halogen-free flame-retardant polymeric composites.</em><br>
            <strong>Journal of Materials Informatics</strong> 2025;5:09.<br>
            DOI: <a href="https://doi.org/10.20517/jmi.2025.09" target="_blank" 
                 style="color:#3f87a6; text-decoration:underline;">
                10.20517/jmi.2025.09
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 开发者信息
    st.markdown("""
    <h2 style="font-size:1.8rem; color:#1e3d59; border-bottom: 2px solid #3f87a6; padding-bottom:0.5rem; margin-bottom:1.5rem;">
        👨💻 开发团队
    </h2>
    """, unsafe_allow_html=True)
    
    col_dev, col_sup = st.columns(2)
    with col_dev:
        st.markdown("""
        <div class="feature-card">
            <p style="font-size:1.1rem;">
                上海大学功能高分子<br>
                PolyDesign <br>
                马维宾 | 李凌 | 张瑜<br>
                宋娜 | 丁鹏
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_sup:
        st.markdown("""
        <div class="feature-card">
            <h3 style="font-size:1.4rem; color:#1e3d59; margin:0 0 1rem 0;">
                🙏 项目支持
            </h3>
            <p style="font-size:1.1rem;">
                云南省科技重点计划<br>
                项目编号：202302AB080022<br>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 登录/注册区域
    st.markdown("""
    <div style="margin-top: 3rem; background: #ffffff; padding: 2rem; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 style="font-size:1.8rem; color:#1e3d59; text-align:center; margin-bottom:1.5rem;">🔐 用户认证</h2>', unsafe_allow_html=True)
    
    tab_login, tab_register, tab_forgot = st.tabs(["登录", "注册", "忘记密码"])

    with tab_login:
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("用户名", key="login_user")
            password = st.text_input("密码", type="password", key="login_pwd")
            
            if st.form_submit_button("登录", use_container_width=True):
                if not all([username, password]):
                    st.error("请输入用户名和密码")
                elif verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            new_user = st.text_input("用户名（4-20位字母数字）", key="reg_user").strip()
            new_pwd = st.text_input("设置密码（至少6位）", type="password", key="reg_pwd")
            confirm_pwd = st.text_input("确认密码", type="password", key="reg_pwd_confirm")
            email = st.text_input("电子邮箱", key="reg_email")
            
            if st.form_submit_button("注册", use_container_width=True):
                if new_pwd != confirm_pwd:
                    st.error("两次密码输入不一致")
                elif len(new_user) < 4 or not new_user.isalnum():
                    st.error("用户名格式不正确")
                elif len(new_pwd) < 6:
                    st.error("密码长度至少6个字符")
                elif "@" not in email:
                    st.error("请输入有效邮箱地址")
                else:
                    if save_user(new_user, new_pwd, email):
                        st.success("注册成功！请登录")
                    else:
                        st.error("用户名已存在")

    with tab_forgot:
        with st.form("forgot_form", clear_on_submit=True):
            email = st.text_input("注册邮箱", key="reset_email")
            new_password = st.text_input("新密码", type="password", key="new_pwd")
            confirm_password = st.text_input("确认密码", type="password", key="confirm_pwd")
            
            if st.form_submit_button("重置密码", use_container_width=True):
                if not all([email, new_password, confirm_password]):
                    st.error("请填写所有字段")
                elif new_password != confirm_password:
                    st.error("两次输入密码不一致")
                elif reset_password_by_email(email, new_password):
                    st.success("密码已重置，请使用新密码登录")
                else:
                    st.error("该邮箱未注册")
    
    st.markdown("</div></div>", unsafe_allow_html=True)  # 结束认证区域和主容器

# --------------------- 主流程控制 ---------------------
if not st.session_state.logged_in:
    show_homepage()
    st.stop()

# --------------------- 预测界面 ---------------------
if st.session_state.logged_in:
    class Predictor:
        def __init__(self, scaler_path, svc_path):
            self.scaler = joblib.load(scaler_path)
            self.model = joblib.load(svc_path)
            
            # 特征列配置
            self.static_cols = ["产品质量指标_Sn%", "添加比例", "一甲%"]
            self.time_series_cols = [
                "黄度值_3min", "6min", "9min", "12min",
                "15min", "18min", "21min", "24min"
            ]
            self.eng_features = [
                'seq_length', 'max_value', 'mean_value', 'min_value',
                'std_value', 'trend', 'range_value', 'autocorr'
            ]
            self.imputer = SimpleImputer(strategy="mean")
    
        def _truncate(self, df):
            time_cols = [col for col in df.columns if "min" in col.lower()]
            time_cols_ordered = [col for col in df.columns if col in time_cols]
            if time_cols_ordered:
                row = df.iloc[0][time_cols_ordered]
                if row.notna().any():
                    max_idx = row.idxmax()
                    max_pos = time_cols_ordered.index(max_idx)
                    for col in time_cols_ordered[max_pos + 1:]:
                        df.at[df.index[0], col] = np.nan
            return df
        
        def _get_slope(self, row, col=None):
            # col 是可选的，将被忽略
            x = np.arange(len(row))
            y = row.values
            mask = ~np.isnan(y)
            if sum(mask) >= 2:
                return stats.linregress(x[mask], y[mask])[0]
            return np.nan
    
        def _calc_autocorr(self, row):
            """计算一阶自相关系数"""
            values = row.dropna().values
            if len(values) > 1:
                n = len(values)
                mean = np.mean(values)
                numerator = sum((values[:-1] - mean) * (values[1:] - mean))
                denominator = sum((values - mean) ** 2)
                if denominator != 0:
                    return numerator / denominator
            return np.nan
    
        def _extract_time_series_features(self, df):
            """修复后的时序特征提取"""
            time_data = df[self.time_series_cols]
            time_data_filled = time_data.ffill(axis=1)
            
            features = pd.DataFrame()
            features['seq_length'] = time_data_filled.notna().sum(axis=1)
            features['max_value'] = time_data_filled.max(axis=1)
            features['mean_value'] = time_data_filled.mean(axis=1)
            features['min_value'] = time_data_filled.min(axis=1)
            features['std_value'] = time_data_filled.std(axis=1)
            features['range_value'] = features['max_value'] - features['min_value']
            features['trend'] = time_data_filled.apply(self._get_slope, axis=1)
            features['autocorr'] = time_data_filled.apply(self._calc_autocorr, axis=1)
            return features
    
        def predict_one(self, sample):
            full_cols = self.static_cols + self.time_series_cols
            df = pd.DataFrame([sample], columns=full_cols)
            df = self._truncate(df)
            
            # 特征合并
            static_features = df[self.static_cols]
            time_features = self._extract_time_series_features(df)
            feature_df = pd.concat([static_features, time_features], axis=1)
            feature_df = feature_df[self.static_cols + self.eng_features]
            
            # 验证维度
            if feature_df.shape[1] != self.scaler.n_features_in_:
                raise ValueError(f"特征维度不匹配！当前：{feature_df.shape[1]}，需要：{self.scaler.n_features_in_}")
            
            X_scaled = self.scaler.transform(feature_df)
            return self.model.predict(X_scaled)[0]

    # 侧边栏主导航
    page = st.sidebar.selectbox(
        "🔧 主功能选择",
        ["性能预测", "配方建议"],
        key="main_nav"
    )

    # 子功能选择（仅在配方建议时显示）
    sub_page = None
    if page == "配方建议":
        sub_page = st.sidebar.selectbox(
            "🔧 子功能选择",
            ["配方优化", "添加剂推荐"],
            key="sub_nav"
        )
    with st.sidebar:
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.logged_in = False  # 设置登录状态为 False
            st.session_state.user = None  # 清除用户信息
            st.rerun()  # 重新加载页面

    @st.cache_resource
    def load_models():
        # 确保模型文件路径正确
        try:
            loi_data = joblib.load("model_and_scaler_loi.pkl")
            ts_data = joblib.load("model_and_scaler_ts1.pkl")
            return {
                "loi_model": loi_data["model"],
                "loi_scaler": loi_data["scaler"],
                "ts_model": ts_data["model"],
                "ts_scaler": ts_data["scaler"],
                "loi_features": ["PP", "AHP", "CFA", "APP", "Pentaerythritol", "DOPO", "ZS", "ZHS", "ZnB"],
                "ts_features": ["PP", "AHP", "CFA", "APP", "Pentaerythritol", "DOPO", "ZS", "ZHS", "ZnB"]
            }
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            return None
    
    models = load_models()
        
    # 获取单位
    def get_unit(fraction_type):
        if fraction_type == "质量":
            return "g"
        elif fraction_type == "质量分数":
            return "wt%"
        elif fraction_type == "体积分数":
            return "vol%"
    
    # 保证PP在首列
    def ensure_pp_first(features):
        if "PP" in features:
            features.remove("PP")
        return ["PP"] + sorted(features)

    # 应用全局样式
    apply_global_styles()
    render_global_header()
      
    if page == "性能预测":
        st.subheader("🔮 性能预测：基于配方预测LOI和TS")
    
        # 初始化 input_values
        if 'input_values' not in st.session_state:
            st.session_state.input_values = {}  # 使用会话状态保存输入值
        
        # 基体材料数据
        matrix_materials = {
            "PP": {"name": "聚丙烯(PP)", "range": (53.5, 99.5)},
            "PA": {"name": "聚酰胺(PA)", "range": (0, 100)},
            "PC/ABS": {"name": "聚碳酸酯/ABS合金", "range": (0, 100)},
            "POM": {"name": "聚甲醛(POM)", "range": (0, 100)},
            "PBT": {"name": "聚对苯二甲酸丁二醇酯(PBT)", "range": (0, 100)},
            "PVC": {"name": "聚氯乙烯(PVC)", "range": (0, 100)},
        }
    
        # 阻燃剂数据
        flame_retardants = {
            "AHP": {"name": "次磷酸铝", "range": (0.0, 25.0)},
            "CFA": {"name": "成炭剂", "range": (0.0, 10.0)},
            "APP": {"name": "聚磷酸铵", "range": (0.0, 19.5)},
            "Pentaerythritol": {"name": "季戊四醇", "range": (0.0, 1.3)},
            "DOPO": {"name": "DOPO阻燃剂", "range": (0.0, 27.0)},
            "ZS": {"name": "锡酸锌", "range": (0.0, 34.5)},
            "ZHS": {"name": "羟基锡酸锌", "range": (0.0, 34.5)},
            "ZnB": {"name": "硼酸锌", "range": (0.0, 2.0)},
        }
    
        # 助剂数据
        additives = {
            "Anti-drip-agent": {"name": "抗滴落剂", "range": (0.0, 0.3)},
            "wollastonite": {"name": "硅灰石", "range": (0.0, 5.0)},
            "SiO2": {"name": "二氧化硅", "range": (0.0, 6.0)},
            "silane coupling agent": {"name": "硅烷偶联剂", "range": (0.5, 3.0)},
            "antioxidant": {"name": "抗氧剂", "range": (0.1, 0.5)},
            "M-2200B": {"name": "润滑剂M-2200B", "range": (0.5, 3.0)},
        }
    
        fraction_type = st.sidebar.selectbox("选择输入的单位", ["质量分数", "质量", "体积分数"])
    
        # 配方成分部分（基体和阻燃剂）
        st.subheader("配方成分")
        
        # 基体选择
        selected_matrix = st.selectbox("选择基体材料", list(matrix_materials.keys()))
        matrix_info = matrix_materials[selected_matrix]
        unit_matrix = get_unit(fraction_type)
        
        # 确保使用浮点数
        matrix_value = st.number_input(
            f"{matrix_info['name']} 含量 ({unit_matrix})", 
            min_value=0.0, 
            max_value=100.0, 
            value=70.0, 
            step=0.1,
            format="%.1f"
        )
        st.session_state.input_values[selected_matrix] = float(matrix_value)
    
        # 阻燃剂选择
        st.subheader("阻燃剂")
        selected_fr = st.multiselect(
            "选择阻燃剂", 
            list(flame_retardants.keys()),
            default=["ZS"]
        )
        
        for fr in selected_fr:
            fr_info = flame_retardants[fr]
            unit_fr = get_unit(fraction_type)
            
            # 确保使用浮点数
            fr_value = st.number_input(
                f"{fr_info['name']} 含量 ({unit_fr})", 
                min_value=0.0, 
                max_value=fr_info["range"][1], 
                value=fr_info["range"][0], 
                step=0.1,
                format="%.1f",
                key=f"fr_{fr}"
            )
            st.session_state.input_values[fr] = float(fr_value)
    
        # 助剂选择
        st.subheader("助剂")
        selected_add = st.multiselect(
            "选择助剂", 
            list(additives.keys()),
            default=["Anti-drip-agent"]
        )
        
        for add in selected_add:
            add_info = additives[add]
            unit_add = get_unit(fraction_type)
            
            # 确保使用浮点数
            add_value = st.number_input(
                f"{add_info['name']} 含量 ({unit_add})", 
                min_value=0.0, 
                max_value=add_info["range"][1], 
                value=add_info["range"][0], 
                step=0.1,
                format="%.1f",
                key=f"add_{add}"
            )
            st.session_state.input_values[add] = float(add_value)
            
        # 校验和预测
        total = sum(st.session_state.input_values.values())  # 总和计算
        is_only_pp = all(v == 0 for k, v in st.session_state.input_values.items() if k != "PP")  # 仅PP配方检查
    
        with st.expander("✅ 输入验证", expanded=True):
            if fraction_type in ["体积分数", "质量分数"]:
                if abs(total - 100.0) > 1e-6:
                    st.error(f"❗ {fraction_type}的总和必须为100%（当前：{total:.2f}%）")
                else:
                    st.success(f"{fraction_type}总和验证通过")
            else:
                st.success("成分总和验证通过")
            
            # 验证配方是否包含锡酸锌或羟基锡酸锌
            if not any(k in ["ZS", "ZHS"] for k in st.session_state.input_values):
                st.error("❗ 配方必须包含锡酸锌(ZS)或羟基锡酸锌(ZHS)。")
            else:
                st.success("配方验证通过，包含锡酸锌或羟基锡酸锌。")
            
            # 验证并点击"开始预测"按钮
            if st.button("🚀 开始预测", type="primary", use_container_width=True):
                # 检查输入总和是否为100%，如果不是则停止
                if fraction_type in ["体积分数", "质量分数"] and abs(total - 100.0) > 1e-6:
                    st.error(f"预测中止：{fraction_type}的总和必须为100%")
                    st.stop()
        
                # 如果是纯PP配方，直接给出模拟值
                if is_only_pp:
                    loi_pred = 17.5
                    ts_pred = 35.0
                else:
                    # 体积分数转换为质量分数
                    if fraction_type == "体积分数":
                        vol_values = np.array(list(st.session_state.input_values.values()))
                        total_mass = vol_values.sum()
                        mass_values = vol_values * total_mass  # 按比例转换
                        st.session_state.input_values = {k: (v / total_mass * 100) for k, v in zip(st.session_state.input_values.keys(), mass_values)}
        
                    # 填充缺失的特征值
                    for feature in models["loi_features"]:
                        if feature not in st.session_state.input_values:
                            st.session_state.input_values[feature] = 0.0
        
                    loi_input = np.array([[st.session_state.input_values[f] for f in models["loi_features"]]])
                    loi_scaled = models["loi_scaler"].transform(loi_input)
                    loi_pred = models["loi_model"].predict(loi_scaled)[0]
        
                    # 处理TS预测
                    for feature in models["ts_features"]:
                        if feature not in st.session_state.input_values:
                            st.session_state.input_values[feature] = 0.0
        
                    ts_input = np.array([[st.session_state.input_values[f] for f in models["ts_features"]]])
                    ts_scaled = models["ts_scaler"].transform(ts_input)
                    ts_pred = models["ts_model"].predict(ts_scaled)[0]
        
                # 显示预测结果
                st.success("预测完成！")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="LOI预测值", value=f"{loi_pred:.2f}%", delta="极限氧指数")
                with col2:
                    st.metric(label="TS预测值", value=f"{ts_pred:.2f} MPa", delta="拉伸强度")
    
    elif page == "配方建议" and sub_page == "添加剂推荐":
        st.subheader("🧪 PVC添加剂智能推荐")
        predictor = Predictor("scaler_fold_1.pkl", "svc_fold_1.pkl")
        
        with st.expander("📋 参考样本", expanded=False):
            # 参考样本数据
            sample_data = [
                {"样本名称": "样本1", "推荐添加剂": "无添加剂", 
                 "Sn%": 19.2, "添加比例": 0.0, "一甲%": 32.0, 
                 "黄度值": [5.36, 6.29, 7.57, 8.57, 10.26, 13.21, 16.54, 27.47]},
                {"样本名称": "样本2", "推荐添加剂": "氯化石蜡", 
                 "Sn%": 18.5, "添加比例": 3.64, "一甲%": 31.05, 
                 "黄度值": [5.29, 6.83, 8.00, 9.32, 11.40, 14.12, 18.37, 30.29]},
                {"样本名称": "样本3", "推荐添加剂": "EA15（市售液体钙锌稳定剂）", 
                 "Sn%": 19.0, "添加比例": 1.04, "一甲%": 31.88, 
                 "黄度值": [5.24, 6.17, 7.11, 8.95, 10.33, 13.21, 17.48, 28.08]}
            ]
            
            for sample in sample_data:
                st.markdown(f"**{sample['样本名称']}** - {sample['推荐添加剂']}")
                cols = st.columns(4)
                cols[0].metric("Sn%", f"{sample['Sn%']}%")
                cols[1].metric("添加比例", f"{sample['添加比例']}%")
                cols[2].metric("一甲%", f"{sample['一甲%']}%")
                
                # 显示黄度值
                yellow_df = pd.DataFrame({
                    "时间(min)": [3, 6, 9, 12, 15, 18, 21, 24],
                    "黄度值": sample['黄度值']
                })
                st.dataframe(yellow_df.set_index("时间(min)"), use_container_width=True)
        
        # 输入表单
        with st.form("additive_form"):
            st.subheader("参数输入")
            
            # 基础参数
            col1, col2, col3 = st.columns(3)
            with col1:
                add_ratio = st.number_input(
                    "添加比例 (%)", 
                    min_value=0.0,
                    max_value=100.0,
                    value=3.64,
                    step=0.1,
                    format="%.2f"
                )
            with col2:
                sn_percent = st.number_input(
                    "Sn含量 (%)", 
                    min_value=0.0, 
                    max_value=100.0,
                    value=18.5,
                    step=0.1,
                    format="%.1f"
                )
            with col3:
                yijia_percent = st.number_input(
                    "一甲含量 (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=31.05,
                    step=0.1,
                    format="%.2f"
                )
            
            # 黄度值
            st.subheader("黄度值随时间变化")
            yellow_cols = st.columns(4)
            yellow_values = {}
            times = [3, 6, 9, 12, 15, 18, 21, 24]
            
            for i, time in enumerate(times):
                with yellow_cols[i % 4]:
                    yellow_values[time] = st.number_input(
                        f"{time}min 黄度值",
                        min_value=0.0,
                        max_value=100.0,
                        value=5.29 + i * 3,
                        step=0.1,
                        format="%.2f",
                        key=f"yellow_{time}"
                    )
            
            submit_btn = st.form_submit_button("🚀 生成推荐方案")
        
        if submit_btn:
            # 构建输入样本
            sample = [
                sn_percent, add_ratio, yijia_percent,
                yellow_values[3], yellow_values[6],
                yellow_values[9], yellow_values[12],
                yellow_values[15], yellow_values[18],
                yellow_values[21], yellow_values[24]
            ]
            
            # 进行预测
            try:
                prediction = predictor.predict_one(sample)
                result_map = {
                    1: "无推荐添加剂", 
                    2: "氯化石蜡", 
                    3: "EA12（脂肪酸复合醇酯）",
                    4: "EA15（液体钙锌稳定剂）", 
                    5: "EA16（环氧化合物）",
                    6: "G70L（多官能团的脂肪酸复合酯混合物）", 
                    7: "EA6（亚磷酸酯）"
                }
                
                additive_name = result_map.get(prediction, "未知类型")
                additive_amount = add_ratio if prediction != 1 else 0.0
                
                # 构建配方表
                formula_data = [
                    ["PVC", 100.00],
                    ["加工助剂ACR", 1.00],
                    ["外滑剂70S", 0.35],
                    ["MBS", 5.00],
                    ["316A", 0.20],
                    ["稳定剂", 1]
                ]
                
                
                # 创建格式化表格
                df = pd.DataFrame(formula_data, columns=["材料名称", "份数（基于PVC 100份）"])
                
                # 展示推荐结果
                st.success("添加剂推荐完成！")
                col_res, col_table = st.columns([1, 2])
                
                with col_res:
                    st.markdown(f"### **在添加剂比例为{(additive_amount:.2f)/100}份时，推荐添加剂种类为**")
                    st.markdown(f"<div style='font-size:24px; color:#3f87a6; font-weight:bold; margin:10px 0;'>{additive_name}</div>", unsafe_allow_html=True)
                
                with col_table:
                    st.markdown("### **完整配方表**")
                    st.dataframe(
                        df, 
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "材料名称": "材料名称",
                            "份数（基于PVC 100份）": st.column_config.NumberColumn(
                                "份数",
                                format="%.4f"
                            )
                        }
                    )
                    
            except Exception as e:
                st.error(f"预测过程中出错: {str(e)}")

    # 添加页脚
    st.markdown("""
    <hr>
    <footer>
        <p>© 2025 阻燃聚合物复合材料智能设计平台</p>
        <p>声明：本平台仅供学术研究、技术验证等非营利性科研活动使用，严禁用于任何商业用途。</p>
    </footer>
    """, unsafe_allow_html=True)
