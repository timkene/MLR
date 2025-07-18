import duckdb
import os
import pandas as pd
import streamlit as st
import polars as pl

# Configure the Streamlit page
st.set_page_config(
    page_title="MLR Analysis - DLT Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("MLR Analysis Dashboard")

@st.cache_data(ttl=3600)
def load_data_from_motherduck():
    """Load data from MotherDuck with caching"""
    try:
        # Get MotherDuck token from environment variables (Railway)
        motherduck_token = os.environ.get("MOTHERDUCK_TOKEN")
        
        if not motherduck_token:
            st.error("❌ 'MOTHERDUCK_TOKEN' not found in environment variables.")
            st.info("Please add MOTHERDUCK_TOKEN to your Railway project variables.")
            return None
        
        # Connect using token in the connection string
        con = duckdb.connect(f"md:my_CIL_DB?motherduck_token={motherduck_token}")


        # Connect using token in the connection string
        con = duckdb.connect(f"md:my_CIL_DB?motherduck_token={motherduck_token}")

        # Step 3: Query the tables and load into pandas DataFrames
        with st.spinner("Loading data from MotherDuck..."):
            GROUP_CONTRACT = con.execute("SELECT * FROM clearline_db.group_contract").fetchdf()
            CLAIMS = con.execute("SELECT * FROM clearline_db.claims").fetchdf()
            GROUPS = con.execute("SELECT * FROM clearline_db.all_group").fetchdf()
            DEBIT = con.execute("SELECT * FROM clearline_db.debit_note").fetchdf()
            PA = con.execute("SELECT * FROM clearline_db.total_pa_procedures").fetchdf()
            ACTIVE_ENROLLEE = con.execute("SELECT * FROM clearline_db.all_active_member").fetchdf()
            M_PLAN = con.execute("SELECT * FROM clearline_db.member_plans").fetchdf()
            G_PLAN = con.execute("SELECT * FROM clearline_db.group_plan").fetchdf()
            PLAN = con.execute("SELECT * FROM clearline_db.plans").fetchdf()
            
            
            # Convert to Polars DataFrames
            GROUP_CONTRACT = pl.from_pandas(GROUP_CONTRACT)
            CLAIMS = pl.from_pandas(CLAIMS)
            GROUPS = pl.from_pandas(GROUPS)
            DEBIT = pl.from_pandas(DEBIT)
            PA = pl.from_pandas(PA)
            ACTIVE_ENROLLEE = pl.from_pandas(ACTIVE_ENROLLEE)
            M_PLAN = pl.from_pandas(M_PLAN)
            G_PLAN = pl.from_pandas(G_PLAN)
            PLAN = pl.from_pandas(PLAN)
            
        con.close()
        return PA, GROUP_CONTRACT, CLAIMS, GROUPS, DEBIT, ACTIVE_ENROLLEE, M_PLAN, G_PLAN, PLAN
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None, None, None, None

def calculate_mlr(PA, GROUP_CONTRACT, CLAIMS, GROUPS, DEBIT):
    """Calculate MLR metrics"""
    try:
        # --- PA MLR ---
        PA = PA.with_columns([
            pl.col('requestdate').cast(pl.Datetime),
            pl.col('granted').cast(pl.Float64, strict=False)
        ])
        group_contract_dates = GROUP_CONTRACT.select(['groupname', 'startdate', 'enddate']).with_columns([
            pl.col('startdate').cast(pl.Datetime),
            pl.col('enddate').cast(pl.Datetime)
        ])
        pa_filtered = PA.join(group_contract_dates, on='groupname', how='inner').filter(
            (pl.col('requestdate') >= pl.col('startdate')) & (pl.col('requestdate') <= pl.col('enddate'))
        )
        PA_mlr = pa_filtered.group_by('groupname').agg(
            pl.col('granted').sum().alias('Total cost')
        )

        # --- CLAIMS MLR ---
        CLAIMS = CLAIMS.with_columns([
            pl.col('approvedamount').cast(pl.Float64),
            pl.col('encounterdatefrom').cast(pl.Datetime),
            pl.col('nhisgroupid').cast(pl.Utf8)
        ])
        GROUPS = GROUPS.with_columns(pl.col('groupid').cast(pl.Utf8))
        claims_with_group = CLAIMS.join(
            GROUPS.select(['groupid', 'groupname']),
            left_on='nhisgroupid', right_on='groupid', how='inner'
        )
        claims_with_dates = claims_with_group.join(
            group_contract_dates, on='groupname', how='inner'
        ).filter(
            (pl.col('encounterdatefrom') >= pl.col('startdate')) & (pl.col('encounterdatefrom') <= pl.col('enddate'))
        )
        claims_mlr = claims_with_dates.group_by('groupname').agg(
            pl.col('approvedamount').sum().alias('Total cost')
        ).sort('Total cost', descending=True)

        # --- DEBIT NOTE (filtered by contract dates) ---
        # Ensure DEBIT is pandas DataFrame for filtering
        if not isinstance(DEBIT, pd.DataFrame):
            DEBIT = DEBIT.to_pandas()
        
        # Convert date column and filter out rows containing "tpa" in description
        DEBIT['from'] = pd.to_datetime(DEBIT['from'])
        CURRENT_DEBIT = DEBIT[~DEBIT['description'].str.contains('tpa', case=False, na=False)]
        
        # Change company_name to groupname for consistency
        CURRENT_DEBIT = CURRENT_DEBIT.rename(columns={'company_name': 'groupname'})
        
        # Convert to polars for joining with contract dates
        current_debit_pl = pl.from_pandas(CURRENT_DEBIT)
        
        # Join with contract dates and filter by contract period
        debit_with_dates = current_debit_pl.join(
            group_contract_dates, on='groupname', how='inner'
        ).filter(
            (pl.col('from') >= pl.col('startdate')) & (pl.col('from') <= pl.col('enddate'))
        )
        
        # Group by company and sum amounts within contract period
        DEBIT_BY_CLIENT = debit_with_dates.group_by('groupname').agg(
            pl.col('amount').sum().alias('amount')
        ).sort('amount', descending=True)

        # --- Merge Results ---
        debit_df = DEBIT_BY_CLIENT.rename({'amount': 'Total cost(DEBIT_BY_CLIENT)'})
        pa_df = PA_mlr.rename({'Total cost': 'Total cost(PA)'}).with_columns(
            (pl.col('Total cost(PA)') * 1.4).round(2).alias('PA40%')
        )
        claims_df = claims_mlr.rename({'Total cost': 'Total cost(claims)'})

        # Calculate PA MLR DataFrame
        pa_merged = debit_df.join(
            pa_df.select(['groupname', 'Total cost(PA)', 'PA40%']),
            on='groupname', how='outer'
        )
        pa_merged = pa_merged.with_columns(
            (pl.col('Total cost(DEBIT_BY_CLIENT)') * 0.10).round(2).alias('commission')
        ).select([
            'groupname',
            'Total cost(DEBIT_BY_CLIENT)',
            'Total cost(PA)',
            'PA40%',
            'commission'
        ])
        pa_merged = pa_merged.with_columns([
            (
                (pl.col('PA40%').fill_null(0) +
                    pl.col('commission').fill_null(0)
                ) / pl.col('Total cost(DEBIT_BY_CLIENT)').fill_null(0) * 100
            ).round(2).alias('MLR(PA) (%)')
        ])

        # Calculate CLAIMS MLR DataFrame
        claims_merged = debit_df.join(
            claims_df.select(['groupname', 'Total cost(claims)']),
            on='groupname', how='outer'
        )
        claims_merged = claims_merged.with_columns(
            (pl.col('Total cost(DEBIT_BY_CLIENT)') * 0.10).round(2).alias('commission')
        ).select([
            'groupname',
            'Total cost(DEBIT_BY_CLIENT)',
            'Total cost(claims)',
            'commission'
        ])
        claims_merged = claims_merged.with_columns([
            (
                (
                    pl.col('Total cost(claims)').fill_null(0) +
                    pl.col('commission').fill_null(0)
                ) / pl.col('Total cost(DEBIT_BY_CLIENT)').fill_null(0) * 100
            ).round(2).alias('MLR(CLAIMS) (%)')
        ])

        # Return both DataFrames
        return pa_merged, claims_merged
        
    except Exception as e:
        st.error(f"Error calculating MLR: {str(e)}")
        return pl.DataFrame(), pl.DataFrame()

def calculate_retail_mlr(PA, ACTIVE_ENROLLEE, M_PLAN, G_PLAN, GROUPS, PLAN):
    try:
        # Ensure consistent data types
        ACTIVE_ENROLLEE = ACTIVE_ENROLLEE.with_columns([
            pl.col("legacycode").cast(pl.Utf8),
            pl.col("memberid").cast(pl.Int64)
        ])

        M_PLAN = M_PLAN.with_columns([
            pl.col("memberid").cast(pl.Int64),
            pl.col("planid").cast(pl.Int64),
            pl.col("iscurrent").cast(pl.Utf8)
        ])

        G_PLAN = G_PLAN.with_columns([
            pl.col("planid").cast(pl.Int64),
            pl.col("groupid").cast(pl.Int64),
            pl.col("individualprice").cast(pl.Float64),
            pl.col("familyprice").cast(pl.Float64),
            pl.col("maxnumdependant").cast(pl.Int64)
        ])

        PA = PA.with_columns([
            pl.col("requestdate").cast(pl.Datetime),
            pl.col("iid").cast(pl.Utf8),
            pl.col("granted").cast(pl.Float64)
        ])

        GROUPS = GROUPS.with_columns([
            pl.col("groupid").cast(pl.Int64),
            pl.col("groupname").cast(pl.Utf8)
        ])

        # Filter current plans
        M_PLANN = M_PLAN.filter(pl.col("iscurrent") == "true")
        PAA = PA.with_columns(pl.col("requestdate").dt.year().alias("year"))

        # Join with group names
        G_PLANN = G_PLAN.join(
            GROUPS.select(['groupid', 'groupname']),
            on='groupid',
            how='left'
        )
        # Filter G_PLANN to only include rows where groupname is 'FAMILY SCHEME' (case-insensitive)
        G_PLANN = G_PLANN.filter(
            pl.col("groupname").str.to_lowercase() == "family scheme"
        )

        # Isolate all unique planid in G_PLANN
        unique_planids = G_PLANN.select("planid").unique()

        # Filter ACTIVE_ENROLLEE to only contain data where their planid is inside the isolated planids of G_PLANN
        # First, ensure ACTIVE_ENROLLEE has planid column (join if necessary)
        if 'planid' in ACTIVE_ENROLLEE.columns:
            ACTIVE_ENROLLEE = ACTIVE_ENROLLEE.drop('planid')
        ACTIVE_ENROLLEE = ACTIVE_ENROLLEE.join(
            M_PLANN.select(['memberid', 'planid']),
            on='memberid',
            how='left'
        )
        # Now filter ACTIVE_ENROLLEE to only those with planid in unique_planids
        ACTIVE_RETAIL = ACTIVE_ENROLLEE.join(
            unique_planids,
            on="planid",
            how="inner"
        )

        # Merge G_PLANN and PLAN to get 'planname' into G_PLANN using 'planid' as common column
        if 'planid' in G_PLANN.columns and 'planid' in PLAN.columns:
            F_GPLAN = G_PLANN.join(
                PLAN.select(['planid', 'planname']),
                on='planid',
                how='left'
            )
        else:
            F_GPLAN = G_PLANN

        # Create a new column 'premium' for each row: (individualprice * countofindividual + countoffamily * familyprice)
        FG_PLANN = F_GPLAN.with_columns(
            (pl.col("individualprice") * pl.col("countofindividual") + pl.col("countoffamily") * pl.col("familyprice")).alias("premium")
        )
        # Calculate total retail premium as the sum of the 'premium' column
        # Group by 'planname' and sum 'premium' for each planname
        total_retail_premium_by_plan = FG_PLANN.group_by("planname").agg(
            pl.col("premium").sum().alias("total_premium")
        )

        # Join with ACTIVE_ENROLLEE
        PA_M = PA.join(
            ACTIVE_ENROLLEE.select(['legacycode', 'memberid']),
            left_on='iid',
            right_on='legacycode',
            how='left'
        )

        # Join with M_PLANN
        PA_MP = PA_M.join(
            M_PLANN.select(['memberid', 'planid']),
            on='memberid',
            how='left'
        )

        # Filter PA to only include rows where groupname is 'family scheme' (case-insensitive)
        PAA = PA_MP.filter(
            pl.col("groupname").str.to_lowercase() == "family scheme"
        )

       # Join PLAN to PAA to get 'planname' into PAA using 'planid'
        if 'planid' in PAA.columns and 'planid' in PLAN.columns:
            PAA = PAA.join(
                PLAN.select(['planid', 'planname']),
                on='planid',
                how='left'
            )

        # Join PAA with ACTIVE_ENROLLEE to get effectivedate and terminationdate for each iid
        if 'iid' in PAA.columns and 'legacycode' in ACTIVE_ENROLLEE.columns:
            PAA = PAA.join(
                ACTIVE_ENROLLEE.select(['legacycode', 'effectivedate', 'terminationdate']),
                left_on='iid',
                right_on='legacycode',
                how='left'
            )

        # Filter PAA to only include claims within the customer's active period
        # This is the key step that was missing in your original code
        if all(col in PAA.columns for col in ['iid', 'planname', 'granted', 'requestdate', 'effectivedate', 'terminationdate']):
            # Filter claims to only those within the customer's active enrollment period
            filtered_PAA = PAA.filter(
                (pl.col('requestdate') >= pl.col('effectivedate')) & 
                (pl.col('requestdate') <= pl.col('terminationdate'))
            )
            
            # Now group by IID and planname, and sum the granted amounts
            grouped_PAA = filtered_PAA.group_by(['iid', 'planname']).agg(
                pl.col('granted').sum().alias('total_cost')
            )
            
            # Select final columns: IID (or legacycode), total_cost, planname
            result_df = grouped_PAA.select(['iid', 'total_cost', 'planname'])
        else:
            grouped_PAA = pl.DataFrame()

        # Group result_df by planname and sum total_cost
        total_cost_by_plan = result_df.group_by('planname').agg(
            pl.col('total_cost').sum().alias('total_cost')
        )

        # Ensure 'planname' is string type in both DataFrames before merging
        if 'planname' in total_retail_premium_by_plan.columns and 'planname' in total_cost_by_plan.columns:
            total_retail_premium_by_plan = total_retail_premium_by_plan.with_columns(
                pl.col('planname').cast(pl.Utf8)
            )
            total_cost_by_plan = total_cost_by_plan.with_columns(
                pl.col('planname').cast(pl.Utf8)
            )
            merged_plan_df = total_retail_premium_by_plan.join(
                total_cost_by_plan,
                on='planname',
                how='left'
            )
        else:
            merged_plan_df = pl.DataFrame()

        return result_df, merged_plan_df

    except Exception as e:
        st.error(f"Error calculating MLR: {str(e)}")
        return pl.DataFrame(), pl.DataFrame()

# Main Streamlit app
if __name__ == "__main__":
    # Load data
    PA, GROUP_CONTRACT, CLAIMS, GROUPS, DEBIT, ACTIVE_ENROLLEE, M_PLAN, G_PLAN, PLAN = load_data_from_motherduck()
    
    if all(df is not None for df in [PA, GROUP_CONTRACT, CLAIMS, GROUPS, DEBIT, ACTIVE_ENROLLEE, M_PLAN, G_PLAN, PLAN]):
        # Calculate MLR
        pa_merged, claims_merged = calculate_mlr(PA, GROUP_CONTRACT, CLAIMS, GROUPS, DEBIT)
        
        if pa_merged.height > 0 or claims_merged.height > 0:
            st.subheader("MLR Analysis Results (PA)")
            if pa_merged.height > 0:
                # Convert to pandas for styling
                pa_df = pa_merged.to_pandas()
                
                # Create a function to highlight rows with MLR > 75%
                def highlight_high_mlr(row):
                    if row['MLR(PA) (%)'] > 75:
                        return ['background-color: #ffcccc; color: red; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                # Apply styling
                styled_pa_df = pa_df.style.apply(highlight_high_mlr, axis=1)
                st.dataframe(styled_pa_df, use_container_width=True)
                
                # Get companies with MLR > 75%
                high_mlr_pa_companies = pa_df[pa_df['MLR(PA) (%)'] > 75]['groupname'].tolist()
            else:
                st.warning("No PA MLR data available to display.")
                high_mlr_pa_companies = []

            st.subheader("MLR Analysis Results (Claims)")
            if claims_merged.height > 0:
                # Convert to pandas for styling
                claims_df = claims_merged.to_pandas()
                
                # Create a function to highlight rows with MLR > 75%
                def highlight_high_mlr_claims(row):
                    if row['MLR(CLAIMS) (%)'] > 75:
                        return ['background-color: #ffcccc; color: red; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                # Apply styling
                styled_claims_df = claims_df.style.apply(highlight_high_mlr_claims, axis=1)
                st.dataframe(styled_claims_df, use_container_width=True)
                
                # Get companies with MLR > 75%
                high_mlr_claims_companies = claims_df[claims_df['MLR(CLAIMS) (%)'] > 75]['groupname'].tolist()
            else:
                st.warning("No Claims MLR data available to display.")
                high_mlr_claims_companies = []
            
            # Display companies with high MLR in a table, ignoring any None values
            st.subheader("🚨 Companies with MLR > 75%")

            # Remove None values from the lists
            high_mlr_pa_companies = [c for c in high_mlr_pa_companies if c is not None]
            high_mlr_claims_companies = [c for c in high_mlr_claims_companies if c is not None]

            if high_mlr_pa_companies or high_mlr_claims_companies:
                import pandas as pd

                # Prepare data for the table
                max_len = max(len(high_mlr_pa_companies), len(high_mlr_claims_companies))
                pa_list = high_mlr_pa_companies + [""] * (max_len - len(high_mlr_pa_companies))
                claims_list = high_mlr_claims_companies + [""] * (max_len - len(high_mlr_claims_companies))

                table_df = pd.DataFrame({
                    "PA MLR > 75%": pa_list,
                    "Claims MLR > 75%": claims_list
                })

                st.dataframe(table_df, use_container_width=True)
                st.success("✅ No companies have MLR > 75%")
        else:
            st.warning("No MLR data available to display.")

        # --- Retail MLR Section ---
            st.subheader("Retail MLR Analysis Results")
            try:
                # Call the retail MLR calculation function
                result_df, merged_plan_df = calculate_retail_mlr(
                    PA, ACTIVE_ENROLLEE, M_PLAN, G_PLAN, GROUPS, PLAN
                )

                # Display result_df
                st.markdown("**Retail MLR - Individual/Plan Breakdown**")
                if result_df is not None and result_df.height > 0:
                    st.dataframe(result_df.to_pandas(), use_container_width=True)
                else:
                    st.info("No retail MLR (result_df) data available.")

                # Display total_retail_premium_by_plan
                st.markdown("**Total Retail Premium by Plan**")
                if merged_plan_df is not None and merged_plan_df.height > 0:
                    st.dataframe(merged_plan_df.to_pandas(), use_container_width=True)
                else:
                    st.info("No retail premium by plan data available.")
            except Exception as e:
                st.error(f"Error displaying retail MLR tables: {str(e)}")
        else:
            st.error("Failed to load required data. Please check your connection and try again.")
    else:
        st.error("Failed to load data. Please check your connection and try again.")
    else:
        st.error("Failed to load data. Please check your connection and try again.")



