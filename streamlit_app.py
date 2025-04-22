import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import plotly.express as px

# Load data
df = pd.read_csv("engineered_macro_data.csv")

# Load model
@st.cache_resource
def load_model():
    model_id = "openchat/openchat-3.5-0106"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

chat = load_model()

countries = ["India", "United States", "China", "Germany", "Brazil"]

st.title("🌍 AI-Powered Economic Forecast Narrator")

year = st.selectbox("Select a Year", df["year"].unique())
col1, col2 = st.columns(2)
country_1 = col1.selectbox("Select Country 1", countries, index=0)
country_2 = col2.selectbox("Select Country 2", countries, index=1)

row = df[df["year"] == year].iloc[0]

def generate_prompt(row, c1, c2):
    return f"""### Instruction:
You are an economic analyst. Compare {c1} and {c2} in {int(row['year'])} using the following data:

{c1}:
- GDP Growth: {row[f'GDP_Growth_{c1}']:.2f}%
- Inflation: {row[f'Inflation_{c1}']:.2f}%
- Inflation Delta: {row[f'Inflation_Delta_{c1}']:.2f} percentage points
- GDP Share: {row[f'GDP_Share_{c1}']:.2f}%

{c2}:
- GDP Growth: {row[f'GDP_Growth_{c2}']:.2f}%
- Inflation: {row[f'Inflation_{c2}']:.2f}%
- Inflation Delta: {row[f'Inflation_Delta_{c2}']:.2f} percentage points
- GDP Share: {row[f'GDP_Share_{c2}']:.2f}%

Global Average Inflation: {row['Global_Inflation_Avg']:.2f}%

### Response:"""

if st.button("🔍 Generate Economic Comparison"):
    with st.spinner("Generating..."):
        prompt = generate_prompt(row, country_1, country_2)
        output = chat(prompt, max_new_tokens=300)[0]["generated_text"]
        st.subheader("📊 Economic Comparison")
        st.write(output)

        # GDP Growth Bar Chart
        st.subheader("📈 GDP Growth Comparison")
        gdp_df = pd.DataFrame({
            "Country": [country_1, country_2],
            "GDP Growth (%)": [row[f"GDP_Growth_{country_1}"], row[f"GDP_Growth_{country_2}"]]
        })
        fig_gdp = px.bar(gdp_df, x="Country", y="GDP Growth (%)", color="Country", text="GDP Growth (%)")
        st.plotly_chart(fig_gdp, use_container_width=True)

        # Inflation Comparison
        st.subheader("📊 Inflation Rate Comparison")
        inflation_df = pd.DataFrame({
            "Country": [country_1, country_2],
            "Inflation (%)": [row[f"Inflation_{country_1}"], row[f"Inflation_{country_2}"]]
        })
        fig_inflation = px.bar(inflation_df, x="Country", y="Inflation (%)", color="Country", text="Inflation (%)")
        st.plotly_chart(fig_inflation, use_container_width=True)

        # GDP Share Pie Chart
        st.subheader("🌍 GDP Share of Selected Countries")
        share_df = pd.DataFrame({
            "Country": [country_1, country_2],
            "Share (%)": [row[f"GDP_Share_{country_1}"], row[f"GDP_Share_{country_2}"]]
        })
        fig_share = px.pie(share_df, names="Country", values="Share (%)", title="Global GDP Share")
        st.plotly_chart(fig_share, use_container_width=True)