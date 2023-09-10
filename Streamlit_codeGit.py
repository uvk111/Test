import streamlit as st
import pandas as pd
import plotly.graph_objs as go

st.set_page_config(layout="wide")
st.title('House Price Prediction Dashboard')

#TASK1: READ FILES
# Source_files_dir = "C:\\Users\\Avinash G\\PycharmProjects\\Home_prediction_streamlit\\ModelOutput\\"
FactorsLookup = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/FactorsLookup.csv')
Initialdata = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/Inputdata_FullDF_selectedcols.csv')
summaryCont_df = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/summaryCont_df.csv')
summaryCont_Treated = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/summaryCont_Treated.csv')
summaryCategorical_df = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/summaryCategorical_df.csv')
summaryCategorical_Treated = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/summaryCategorical_Treated.csv')
bivariate_analysis = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/bivariate_analysis.csv')
Correlation_VariableSelection = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/Correlation_VariableSelection.csv')
Regression_Model_Metrics = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/Regression_Model_Metrics.csv')

Scored_Luxury = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/LUXURYMODEL_Scored_Dataframe.csv')
Scored_MidRange = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/MIDRANGEMODEL_Scored_Dataframe.csv')
Scored_Affordable = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/AFFORDABLEMODEL_Scored_Dataframe.csv')


#TASK2:
st.write("")
st.markdown("""
The Home Price Prediction Model is developed by sourcing data from below government websites released under Open Government Licence (OGL)

Home Price January 2023 to July 2023:
https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads

Consumer price inflation January 2023 to July 2023:
https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l55o/mm23

National Statistics Postcode Lookup
https://geoportal.statistics.gov.uk/datasets/national-statistics-postcode-lookup-2021-census-august-2023/about

nomis Housing, Work and Travel, Education, Health bulk
https://www.nomisweb.co.uk/sources/census_2021_bulk
""")

st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 24px; margin: 0; padding: 0;'>Regression Analysis Variables</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Dependent Variable:</h1>", unsafe_allow_html=True)
st.markdown("""
Price
""")

st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Continous Variables:</h1>", unsafe_allow_html=True)
st.markdown("""
InflationRate, NumResidents_Heath_VeryGood, NumResidents_Heath_Good, NumResidents_Heath_Fair, NumResidents_Heath_Bad, NumResidents_Heath_VeryBad, Acctype_Detached,
Acctype_SemiDetached, Acctype_Terraced, Acctype_Flats, Acctype_SharedHouse, Acctype_Converted, Acctype_CommercialBuilding, Acctype_Caravan, HH_noVehicles, HH_EQ1Vehicles,
HH_EQ2Vehicles, HH_GE3Vehicles, HH_noCentralHeating, HH_CentralHeating_GasOnly, HH_CentralHeating_TankGasOnly, HH_CentralHeating_ElectricOnly, HH_CentralHeating_OilOnly,
HH_CentralHeating_WoodOnly, HH_CentralHeating_SoildFuelOnly, HH_CentralHeating_RenewableOnly, HH_CentralHeating_CommunalOnly, HH_CentralHeating_Others, HH_CentralHeating_GT2types_NoRenew,
HH_CentralHeating_GT2types_WithRenew, Num_With1BedRooms, Num_With2BedRooms, Num_With3BedRooms, Num_WithGE4BedRooms, Num_With1Room, Num_With2Rooms, Num_With3Rooms, Num_With4Rooms,
Num_With5Rooms, Num_With6Rooms, Num_With7Rooms, Num_With8Rooms, Num_WithGE9Rooms, NumProperty_GE2RatingBedRooms, NumProperty_1RatingBedRooms, NumProperty_0RatingBedRooms,
NumProperty_Minus1RatingBedRooms, NumProperty_LEMinus2RatingBedRooms, NumProperty_GE2RatingRooms, NumProperty_1RatingRooms, NumProperty_0RatingRooms, NumProperty_Minus1RatingRooms,
NumProperty_LEMinus2RatingRooms, Num_HH_OwnedOutright, Num_HH_OwnedMortgage, Num_HH_SharedOwnership, Num_HH_CouncilRents, Num_HH_PrivateRented, Num_HH_OtherPrivateRented, Num_HH_LivesRentFree,
NumProperty_NoSecAddr, NumProperty_UKSecAddr, NumProperty_OutsideUKSecAddr, Occupation_SrOfficials, Occupation_Professionals, Occupation_AssoProfessionals, Occupation_Administrative, Occupation_SkillTrades,
Occupation_Carer, Occupation_SalesService, Occupation_MachineOperatives, Occupation_Elementary, Unemployment_LT12Months, Unemployment_Exactly12Months, Unemployment_NeverWorked, Num_NHS_GenHosp, Num_NHS_MentalHosp,
Num_NHS_OtherHosp, Num_LocalAuthChildernHome, Num_LocalAuthCareHomeWithNursing, Num_LocalAuthCareHomeWithOutNursing, Num_SocialHostels, Num_OtherCareHomeWithNursing, Num_OtherCareHomeWithOutNursing, Num_OtherChildrenHomeSecure,
Num_OtherMentalHospSecure, Num_DefenceBuildings, Num_PrisonServiceBuildings, Num_BailHostels, Num_DetentionCentres, Num_OtherEducationBuildings, Num_OtherYouthHostels, Num_OtherHomelessHostels,
Num_HolidayAccomodations, Num_ReligiousBuildings, Num_StaffAccomodations, Industries_Agri, Industries_Mining, Industries_Manufacturing, Industries_PowerStations, Industries_WaterMgmt,
Industries_Construction, Industries_Wholsale, Industries_Transportation, Industries_Accomodations, Industries_Communication, Industries_FinancialInst, Industries_RealEstate, Industries_Professional,
Industries_Administrative, Industries_Defence, Industries_Education, Industries_SocialWork, Industries_Others,
""")

st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Categorical Variables:</h1>", unsafe_allow_html=True)
st.markdown("""
Property_Type, New_Property, Duration_Relates_to_the_tenure, MonthYear
""")
st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)

#TASK3: EXPLORATORY DATA ANALYSIS
#TASK3.1: UNIVARIATE ANALYSIS - CONTINUOUS DATA
st.markdown("<h1 style='font-size: 24px; margin: 0; padding: 0;'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
st.markdown("""""")
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Univariate Analysis - Continous Variables</h1>", unsafe_allow_html=True)
st.markdown("""""")
selected_factors = st.selectbox('Please select Home based factor', FactorsLookup['Factor'].unique())
OriginalVariables = FactorsLookup[FactorsLookup['Factor'] == selected_factors]['OriginalVariable'].unique()
TreatedVariables = FactorsLookup[FactorsLookup['Factor'] == selected_factors]['TreatedVariable'].unique()

Orig = summaryCont_df[summaryCont_df['Variable'].isin(OriginalVariables)]
Orig = Orig.set_index('Variable', drop=True)
Treated = summaryCont_Treated[summaryCont_Treated['Variable'].isin(TreatedVariables)]
Treated = Treated.set_index('Variable', drop=True)

st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Variables Before Treatment:</h1>", unsafe_allow_html=True)
st.dataframe(Orig, use_container_width=True)
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Variables After Treatment:</h1>", unsafe_allow_html=True)
st.dataframe(Treated, use_container_width=True)
st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)
#TASK3.2: UNIVARIATE ANALYSIS - CATEGORICAL DATA
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Univariate Analysis - Categorical Variables</h1>", unsafe_allow_html=True)
st.markdown("""""")

# summaryCategorical_df
summaryCategorical_df = summaryCategorical_df.set_index('Variable Name', drop=True)
# summaryCategorical_Treated
summaryCategorical_Treated = summaryCategorical_Treated.set_index('Variable Name', drop=True)

st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Variables Before Treatment:</h1>", unsafe_allow_html=True)
st.dataframe(summaryCategorical_df, use_container_width=True)
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Variables After Treatment:</h1>", unsafe_allow_html=True)
st.dataframe(summaryCategorical_Treated, use_container_width=True)

st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)

#BIVARIATE ANALYSIS
st.markdown("""""")
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Bivariate Analysis - Categorical Variables</h1>", unsafe_allow_html=True)
st.markdown("""""")
selected_Segment = st.selectbox('Please select Home Segment', bivariate_analysis['Segment'].unique())
Filter_bivardf = bivariate_analysis[bivariate_analysis['Segment'] == selected_Segment]

Property_Type_Filter = Filter_bivardf[Filter_bivardf['Variable_Name'] == 'Property_Type']
New_Property_Filter = Filter_bivardf[Filter_bivardf['Variable_Name'] == 'New_Property']
Duration_Relates_to_the_tenure_Filter = Filter_bivardf[Filter_bivardf['Variable_Name'] == 'Duration_Relates_to_the_tenure']
MonthYear_Filter = Filter_bivardf[Filter_bivardf['Variable_Name'] == 'MonthYear']

#PropertyType
trace11 = go.Bar(x=Property_Type_Filter['Values'], y=Property_Type_Filter['Count'], name="Number of Homes", marker=dict(color='cornflowerblue'))
trace12 = go.Scatter(x=Property_Type_Filter['Values'], y=Property_Type_Filter['AvgDV'], mode='lines', name="AvgDV", yaxis='y2', line=dict(color='red'))
layout = go.Layout(title=dict(text="Number of Homes vs Dependent Variable by Property Type", font=dict(size=16)), xaxis=dict(title="Property Type", tickfont=dict(size=12)), yaxis=dict(title="Number of Homes", tickfont=dict(size=12), range=[0, Property_Type_Filter['Count'].max()], tick0=0, dtick=100000, showgrid=False), yaxis2=dict(title="Average Dependent Variable", tickfont=dict(size=12), overlaying='y', side='right', range=[0, Property_Type_Filter['AvgDV'].max()], tick0=0, dtick=100000, showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=True, legend=dict(orientation='v', x=1, y=1.2))
Chart1_1 = go.Figure(data=[trace11, trace12], layout=layout)

#New_Property
trace21 = go.Bar(x=New_Property_Filter['Values'], y=New_Property_Filter['Count'], name="Number of Homes", marker=dict(color='cornflowerblue'))
trace22 = go.Scatter(x=New_Property_Filter['Values'], y=New_Property_Filter['AvgDV'], mode='lines', name="AvgDV", yaxis='y2', line=dict(color='red'))
layout = go.Layout(title=dict(text="Number of Homes vs Dependent Variable by New Property", font=dict(size=16)), xaxis=dict(title="New Property", tickfont=dict(size=12)), yaxis=dict(title="Number of Homes", tickfont=dict(size=12), range=[0, New_Property_Filter['Count'].max()], tick0=0, dtick=100000, showgrid=False), yaxis2=dict(title="Average Dependent Variable", tickfont=dict(size=12), overlaying='y', side='right', range=[0, New_Property_Filter['AvgDV'].max()], tick0=0, dtick=100000, showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=True, legend=dict(orientation='v', x=1, y=1.2))
Chart1_2 = go.Figure(data=[trace21, trace22], layout=layout)

#Duration_Relates_to_the_tenure
trace31 = go.Bar(x=Duration_Relates_to_the_tenure_Filter['Values'], y=Duration_Relates_to_the_tenure_Filter['Count'], name="Number of Homes", marker=dict(color='cornflowerblue'))
trace32 = go.Scatter(x=Duration_Relates_to_the_tenure_Filter['Values'], y=Duration_Relates_to_the_tenure_Filter['AvgDV'], mode='lines', name="AvgDV", yaxis='y2', line=dict(color='red'))
layout = go.Layout(title=dict(text="Number of Homes vs Dependent Variable by Tenure", font=dict(size=16)), xaxis=dict(title="Duration Related to Tenure", tickfont=dict(size=12)), yaxis=dict(title="Number of Homes", tickfont=dict(size=12), range=[0, Duration_Relates_to_the_tenure_Filter['Count'].max()], tick0=0, dtick=100000, showgrid=False), yaxis2=dict(title="Average Dependent Variable", tickfont=dict(size=12), overlaying='y', side='right', range=[0, Duration_Relates_to_the_tenure_Filter['AvgDV'].max()], tick0=0, dtick=100000, showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=True, legend=dict(orientation='v', x=1, y=1.2))
Chart1_3 = go.Figure(data=[trace31, trace32], layout=layout)

#MonthYear
trace41 = go.Bar(x=MonthYear_Filter['Values'], y=MonthYear_Filter['Count'], name="Number of Loans", marker=dict(color='cornflowerblue'))
trace42 = go.Scatter(x=MonthYear_Filter['Values'], y=MonthYear_Filter['AvgDV'], mode='lines', name="AvgDV", yaxis='y2', line=dict(color='red'))
layout = go.Layout(title=dict(text="Number of Homes vs Dependent Variable by Months", font=dict(size=16)), xaxis=dict(title="Time Period (In Month Year)", tickfont=dict(size=12)), yaxis=dict(title="Number of Homes", tickfont=dict(size=12), range=[0, MonthYear_Filter['Count'].max()], tick0=0, dtick=100000, showgrid=False), yaxis2=dict(title="Average Dependent Variable", tickfont=dict(size=12), overlaying='y', side='right', range=[0, MonthYear_Filter['AvgDV'].max()], tick0=0, dtick=100000, showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=True, legend=dict(orientation='v', x=1, y=1.2))
Chart1_4 = go.Figure(data=[trace41, trace42], layout=layout)

left_column1, right_column1 = st.columns([2, 2])
with left_column1:
    st.plotly_chart(Chart1_1, use_container_width=True)
    st.plotly_chart(Chart1_2, use_container_width=True)
with right_column1:
    st.plotly_chart(Chart1_3, use_container_width=True)
    st.plotly_chart(Chart1_4, use_container_width=True)

st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)

#Segmentation
st.markdown("""""")
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>K-Means Clustering Segmentation</h1>", unsafe_allow_html=True)
st.markdown("""""")
Aggdata2_1 = Initialdata.groupby('HomeSegment').agg(Count=('HomeSegment', 'size'), AvgDV=('Avg_home_price', 'mean')).reset_index()
trace21 = go.Bar(x=Aggdata2_1['HomeSegment'], y=Aggdata2_1['Count'], name="Number of Homes", marker=dict(color='cornflowerblue'))
trace22 = go.Scatter(x=Aggdata2_1['HomeSegment'], y=Aggdata2_1['AvgDV'], mode='lines', name="AvgDV", yaxis='y2', line=dict(color='red'))
layout = go.Layout(title=dict(text="Number of Homes vs Dependent Variable by Home Segments", font=dict(size=16)), xaxis=dict(title="K-Means Algorithm Home Segments", tickfont=dict(size=12)), yaxis=dict(title="Number of Homes", tickfont=dict(size=12), range=[0, Property_Type_Filter['Count'].max()], tick0=0, dtick=50000, showgrid=False), yaxis2=dict(title="Average Dependent Variable", tickfont=dict(size=12), overlaying='y', side='right', range=[0, Property_Type_Filter['AvgDV'].max()], tick0=0, dtick=100000, showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=True, legend=dict(orientation='v', x=1, y=1.2))
Chart2_1 = go.Figure(data=[trace21, trace22], layout=layout)

left_column2, right_column2 = st.columns([2, 2])
with left_column2:
    st.plotly_chart(Chart2_1, use_container_width=True)
with right_column2:
    st.write(Aggdata2_1)

st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)

st.markdown("""""")
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Correlation Analysis - Feature Selection</h1>", unsafe_allow_html=True)
st.markdown("""""")

Affordable_Varlist = ['InflationRate_T', 'NumResidents_Heath_VeryGood_T', 'NumResidents_Heath_Fair_T', 'NumResidents_Heath_Bad_T', 'NumResidents_Heath_VeryBad_T', 'Acctype_Detached_T', 'Acctype_Terraced_T', 'Acctype_Flats_T', 'Acctype_Caravan_T', 'HH_noVehicles_T', 'HH_EQ1Vehicles_T', 'HH_EQ2Vehicles_T', 'HH_GE3Vehicles_T', 'HH_noCentralHeating_T', 'HH_CentralHeating_GasOnly_T', 'HH_CentralHeating_TankGasOnly_T', 'HH_CentralHeating_OilOnly_T', 'HH_CentralHeating_WoodOnly_T', 'HH_CentralHeating_RenewableOnly_T', 'HH_CentralHeating_Others_T', 'HH_CentralHeating_GT2types_WithRenew_T', 'Num_With1BedRooms_T', 'Num_With2BedRooms_T', 'Num_With3BedRooms_T', 'Num_WithGE4BedRooms_T', 'Num_With2Rooms_T', 'Num_With3Rooms_T', 'Num_With4Rooms_T', 'Num_With6Rooms_T', 'Num_With7Rooms_T', 'Num_With8Rooms_T', 'Num_WithGE9Rooms_T', 'NumProperty_GE2RatingBedRooms_T', 'NumProperty_1RatingBedRooms_T', 'NumProperty_0RatingBedRooms_T', 'NumProperty_Minus1RatingBedRooms_T', 'NumProperty_GE2RatingRooms_T', 'NumProperty_1RatingRooms_T', 'NumProperty_0RatingRooms_T', 'NumProperty_Minus1RatingRooms_T', 'Num_HH_OwnedOutright_T', 'Num_HH_OwnedMortgage_T', 'Num_HH_CouncilRents_T', 'Num_HH_PrivateRented_T', 'Num_HH_LivesRentFree_T', 'NumProperty_OutsideUKSecAddr_T', 'Occupation_SrOfficials_T', 'Occupation_Professionals_T', 'Occupation_AssoProfessionals_T', 'Occupation_Administrative_T', 'Occupation_Carer_T', 'Occupation_SalesService_T', 'Occupation_MachineOperatives_T', 'Occupation_Elementary_T', 'Unemployment_Exactly12Months_T', 'Unemployment_NeverWorked_T', 'Industries_Agri_T', 'Industries_Manufacturing_T', 'Industries_Construction_T', 'Industries_Wholsale_T', 'Industries_Transportation_T', 'Industries_Communication_T', 'Industries_FinancialInst_T', 'Industries_RealEstate_T', 'Industries_Professional_T', 'Industries_Education_T', 'Industries_SocialWork_T', 'Industries_Others_T', 'Property_Type_D', 'Property_Type_T', 'Property_Type_F', 'Property_Type_O', 'Duration_Relates_to_the_tenure_F', 'Duration_Relates_to_the_tenure_L', 'InflationRate_T']
MidRange_Varlist = ['InflationRate_T', 'NumResidents_Heath_VeryGood_T', 'NumResidents_Heath_Good_T', 'NumResidents_Heath_Fair_T', 'NumResidents_Heath_Bad_T', 'NumResidents_Heath_VeryBad_T', 'Acctype_Detached_T', 'Acctype_Terraced_T', 'HH_noVehicles_T', 'HH_EQ1Vehicles_T', 'HH_EQ2Vehicles_T', 'HH_GE3Vehicles_T', 'HH_noCentralHeating_T', 'HH_CentralHeating_TankGasOnly_T', 'HH_CentralHeating_OilOnly_T', 'HH_CentralHeating_RenewableOnly_T', 'HH_CentralHeating_Others_T', 'HH_CentralHeating_GT2types_WithRenew_T', 'Num_With1BedRooms_T', 'Num_With2BedRooms_T', 'Num_WithGE4BedRooms_T', 'Num_With2Rooms_T', 'Num_With3Rooms_T', 'Num_With4Rooms_T', 'Num_With5Rooms_T', 'Num_With6Rooms_T', 'Num_With7Rooms_T', 'Num_With8Rooms_T', 'Num_WithGE9Rooms_T', 'NumProperty_GE2RatingBedRooms_T', 'NumProperty_1RatingBedRooms_T', 'NumProperty_0RatingBedRooms_T', 'NumProperty_GE2RatingRooms_T', 'NumProperty_1RatingRooms_T', 'NumProperty_0RatingRooms_T', 'Num_HH_OwnedOutright_T', 'Num_HH_OwnedMortgage_T', 'Num_HH_CouncilRents_T', 'Num_HH_PrivateRented_T', 'NumProperty_NoSecAddr_T', 'NumProperty_OutsideUKSecAddr_T', 'Occupation_SrOfficials_T', 'Occupation_Professionals_T', 'Occupation_AssoProfessionals_T', 'Occupation_Administrative_T', 'Occupation_Carer_T', 'Occupation_SalesService_T', 'Occupation_MachineOperatives_T', 'Occupation_Elementary_T', 'Unemployment_Exactly12Months_T', 'Industries_Agri_T', 'Industries_Manufacturing_T', 'Industries_Construction_T', 'Industries_Communication_T', 'Industries_FinancialInst_T', 'Industries_RealEstate_T', 'Industries_Professional_T', 'Industries_Administrative_T', 'Industries_Education_T', 'Industries_Others_T', 'Property_Type_D', 'Property_Type_T', 'Property_Type_F', 'Property_Type_O', 'Duration_Relates_to_the_tenure_F', 'Duration_Relates_to_the_tenure_L', 'InflationRate_T']
Luxury_Varlist = ['InflationRate_T', 'NumResidents_Heath_VeryGood_T', 'NumResidents_Heath_Fair_T', 'NumResidents_Heath_Bad_T', 'Acctype_Detached_T', 'Acctype_Flats_T', 'Acctype_SharedHouse_T', 'HH_noVehicles_T', 'HH_EQ1Vehicles_T', 'HH_GE3Vehicles_T', 'HH_noCentralHeating_T', 'HH_CentralHeating_ElectricOnly_T', 'HH_CentralHeating_GT2types_WithRenew_T', 'Num_With1BedRooms_T', 'Num_With2BedRooms_T', 'Num_WithGE4BedRooms_T', 'Num_With2Rooms_T', 'Num_With3Rooms_T', 'Num_With4Rooms_T', 'Num_With6Rooms_T', 'Num_With7Rooms_T', 'Num_With8Rooms_T', 'Num_WithGE9Rooms_T', 'NumProperty_GE2RatingBedRooms_T', 'NumProperty_1RatingBedRooms_T', 'NumProperty_0RatingBedRooms_T', 'NumProperty_Minus1RatingBedRooms_T', 'NumProperty_GE2RatingRooms_T', 'NumProperty_1RatingRooms_T', 'NumProperty_0RatingRooms_T', 'Num_HH_OwnedOutright_T', 'Num_HH_SharedOwnership_T', 'Num_HH_PrivateRented_T', 'NumProperty_OutsideUKSecAddr_T', 'Occupation_SrOfficials_T', 'Occupation_Professionals_T', 'Occupation_SkillTrades_T', 'Occupation_Carer_T', 'Occupation_SalesService_T', 'Occupation_MachineOperatives_T', 'Occupation_Elementary_T', 'Unemployment_Exactly12Months_T', 'Industries_Mining_T', 'Industries_Manufacturing_T', 'Industries_PowerStations_T', 'Industries_WaterMgmt_T', 'Industries_Wholsale_T', 'Industries_Transportation_T', 'Industries_Accomodations_T', 'Industries_Communication_T', 'Industries_FinancialInst_T', 'Industries_RealEstate_T', 'Industries_Professional_T', 'Industries_Defence_T', 'Industries_Education_T', 'Industries_SocialWork_T', 'Industries_Others_T', 'Property_Type_D', 'Property_Type_F', 'Duration_Relates_to_the_tenure_F', 'Duration_Relates_to_the_tenure_L', 'InflationRate_T']

selected_CorrSegment = st.selectbox('Please select Home Segment', Correlation_VariableSelection['segment'].unique())
Filter_bivardf = Correlation_VariableSelection[Correlation_VariableSelection['segment'] == selected_CorrSegment]
Filter_bivardf = Filter_bivardf[Filter_bivardf['VariableA'] == 'DV']

if selected_CorrSegment == 'Affordable':
    Filter_bivardf = Filter_bivardf[Filter_bivardf['VariableB'].isin(Affordable_Varlist)]
elif selected_CorrSegment == 'Mid_Range':
    Filter_bivardf = Filter_bivardf[Filter_bivardf['VariableB'].isin(MidRange_Varlist)]
elif selected_CorrSegment == 'Luxury':
    Filter_bivardf = Filter_bivardf[Filter_bivardf['VariableB'].isin(Luxury_Varlist)]
else:
    Filter_bivardf

Filter_bivardf = Filter_bivardf.set_index('segment', drop=True)
st.write(Filter_bivardf)

st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)

st.markdown("""""")
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Model Evaluation</h1>", unsafe_allow_html=True)
st.markdown("""""")
Segment_selected = st.selectbox('Please select Home Segment', Regression_Model_Metrics['Segment'].unique())
MevalMetric = Regression_Model_Metrics[Regression_Model_Metrics['Segment'] == Segment_selected]

trace41 = go.Bar(x=MevalMetric['Model Name'], y=MevalMetric['R2'], name="R2", marker=dict(color='tomato'))
layout = go.Layout(title=dict(text="Model Evaluation - R2", font=dict(size=16)), xaxis=dict(title="Model Name", tickfont=dict(size=12)), yaxis=dict(title="R2", tickfont=dict(size=12), showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=False, legend=dict(orientation='v', x=1, y=1.2))
Chart4_1 = go.Figure(data=[trace41], layout=layout)

trace42 = go.Bar(x=MevalMetric['Model Name'], y=MevalMetric['RMSE'], name="RMSE", marker=dict(color='tan'))
layout = go.Layout(title=dict(text="Model Evaluation - RMSE", font=dict(size=16)), xaxis=dict(title="Model Name", tickfont=dict(size=12)), yaxis=dict(title="RMSE", tickfont=dict(size=12), showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=False, legend=dict(orientation='v', x=1, y=1.2))
Chart4_2 = go.Figure(data=[trace42], layout=layout)

trace43 = go.Bar(x=MevalMetric['Model Name'], y=MevalMetric['MSE'], name="MSE", marker=dict(color='steelblue'))
layout = go.Layout(title=dict(text="Model Evaluation - MSE", font=dict(size=16)), xaxis=dict(title="Model Name", tickfont=dict(size=12)), yaxis=dict(title="MSE", tickfont=dict(size=12), showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=False, legend=dict(orientation='v', x=1, y=1.2))
Chart4_3 = go.Figure(data=[trace43], layout=layout)

trace44 = go.Bar(x=MevalMetric['Model Name'], y=MevalMetric['MAE'], name="MAE", marker=dict(color='lightseagreen'))
layout = go.Layout(title=dict(text="Model Evaluation - MAE", font=dict(size=16)), xaxis=dict(title="Model Name", tickfont=dict(size=12)), yaxis=dict(title="MAE", tickfont=dict(size=12), showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=False, legend=dict(orientation='v', x=1, y=1.2))
Chart4_4 = go.Figure(data=[trace44], layout=layout)

left_column1, right_column1 = st.columns([2, 2])
with left_column1:
    st.plotly_chart(Chart4_1, use_container_width=True)
    st.plotly_chart(Chart4_2, use_container_width=True)
with right_column1:
    st.plotly_chart(Chart4_3, use_container_width=True)
    st.plotly_chart(Chart4_4, use_container_width=True)

MevalMetric = MevalMetric.set_index('Model Name', drop=True)
st.write(MevalMetric)
st.markdown("Note: The scale of Dependent variable(Price of Home) is between Thousands and Millions")
st.markdown("The Error residue is computed based on actual(Price - predicted(Y-Predicted). ")
st.markdown("So the error Matrix Y-axis scale in chart will be high")
st.markdown("As a thumb rule lower the error better the model")
st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)

st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>End of Report</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)
