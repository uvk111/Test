import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff

st.set_page_config(layout="wide")
st.title('House Price Prediction Dashboard')

#TASK1: READ FILES
# Source_files_dir = "C:\\Users\\Avinash G\\PycharmProjects\\Home_prediction_streamlit\\ModelOutput\\"
FactorsLookup = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/FactorsLookup.csv')
Initialdata = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/Inputdata_Reduced.csv')
summaryCont_df = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/Continuous_BeforeEDA.csv')
summaryCont_Treated = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/Continuous_AfterEDA.csv')
summaryCategorical_df = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/Categorical_BeforeEDA.csv')
summaryCategorical_Treated = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/Categorical_AfterEDA.csv')
bivariate_analysis = pd.read_csv('https://raw.githubusercontent.com/uvk111/Testing_pipeline/main/data/BivariateResults.csv')
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



#Segmentation
st.markdown("""""")
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Home Price Segmentation using K-Means</h1>", unsafe_allow_html=True)
st.markdown("""""")
Aggdata2_1 = Initialdata.groupby('HomeSegment').agg(Count=('HomeSegment', 'size'), AvgDV=('Avg_home_price', 'mean')).reset_index()
Aggdata2_1 = Aggdata2_1.sort_values(by='AvgDV', ascending=True)
# trace21 = go.Bar(x=Aggdata2_1['HomeSegment'], y=Aggdata2_1['Count'], name="Number of Homes", marker=dict(color='cornflowerblue'))
# trace22 = go.Scatter(x=Aggdata2_1['HomeSegment'], y=Aggdata2_1['AvgDV'], mode='lines', name="AvgDV", yaxis='y2', line=dict(color='red'))
# layout = go.Layout(title=dict(text="Number of Homes vs Dependent Variable by Home Segments", font=dict(size=16)), xaxis=dict(title="K-Means Algorithm Home Segments", tickfont=dict(size=12)), yaxis=dict(title="Number of Homes", tickfont=dict(size=12), range=[0, Property_Type_Filter['Count'].max()], tick0=0, dtick=50000, showgrid=False), yaxis2=dict(title="Average Dependent Variable", tickfont=dict(size=12), overlaying='y', side='right', range=[0, Property_Type_Filter['AvgDV'].max()], tick0=0, dtick=100000, showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=True, legend=dict(orientation='v', x=1, y=1.2))
# Chart2_1 = go.Figure(data=[trace21, trace22], layout=layout)

pie_colors = ["olivedrab", "salmon", "mediumslateblue"]
trace22 = go.Pie(labels=Aggdata2_1['HomeSegment'],values=Aggdata2_1['Count'],hoverinfo='label+percent',marker=dict(colors=pie_colors,line=dict(color='white',width=0)))
# layout = go.Layout(title='Pie Chart')
Chart2_1 = go.Figure(data=[trace22])
# , layout=layout)

left_column2, right_column2 = st.columns([2, 2])
with left_column2:
    st.plotly_chart(Chart2_1, use_container_width=True)
with right_column2:
    st.write(Aggdata2_1)

st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)

#BIVARIATE ANALYSIS
st.markdown("""""")
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Bivariate Analysis - Categorical Variables</h1>", unsafe_allow_html=True)
st.markdown("""""")
selected_Segment = st.selectbox('Please select Home Segment', bivariate_analysis['HomeSegment'].unique())
Filter_bivardf = bivariate_analysis[bivariate_analysis['HomeSegment'] == selected_Segment]

Property_Type_Filter = Filter_bivardf[Filter_bivardf['Variable_Name'] == 'Property_Type']
New_Property_Filter = Filter_bivardf[Filter_bivardf['Variable_Name'] == 'New_Property']
Duration_Relates_to_the_tenure_Filter = Filter_bivardf[Filter_bivardf['Variable_Name'] == 'Duration_Relates_to_the_tenure']
MonthYear_Filter = Filter_bivardf[Filter_bivardf['Variable_Name'] == 'MonthYear']

#PropertyType
custom_labels = {"D": "Detached", "F": "Flats/Maisonettes", "T": "Terraced", "S": "Semi-Detached", "O": "Other"}
bar_colors = ["darkkhaki", "aquamarine", "lavender", "skyblue", "mistyrose"]
trace11 = go.Bar(x=[custom_labels[val] for val in Property_Type_Filter['Unique_Values']], y=Property_Type_Filter['Count'], name="Number of Homes", marker=dict(color=bar_colors), text=[count for count in Property_Type_Filter['Count']], showlegend=False)
trace12 = go.Scatter(x=[custom_labels[val] for val in Property_Type_Filter['Unique_Values']], y=Property_Type_Filter['AvgDV'], mode='lines', name="AvgDV", yaxis='y2', line=dict(color='red'))
layout = go.Layout(title=dict(text="Home Price by Property Type", font=dict(size=16), y=0.99), xaxis=dict(title="Property Type", tickfont=dict(size=12)), yaxis=dict(title="Number of Homes",tickfont=dict(size=12), range=[0, Property_Type_Filter['Count'].max()], tick0=0, dtick=100000, showgrid=False), yaxis2=dict(title="Average Dependent Variable", tickfont=dict(size=12), overlaying='y', side='right', range=[0, Property_Type_Filter['AvgDV'].max()], tick0=0, dtick=100000, showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=True, legend=dict(orientation='v', x=1, y=1.2))
Chart1_1 = go.Figure(data=[trace11, trace12], layout=layout)

#New_Property
custom_labels = {"Y": "Newly Built Property", "N": "Established Residential Building"}
bar_colors = ["thistle", "teal"]
trace21 = go.Bar(x=[custom_labels[val] for val in New_Property_Filter['Unique_Values']], y=New_Property_Filter['Count'], name="Number of Homes", marker=dict(color=bar_colors), text=[count for count in New_Property_Filter['Count']], showlegend=False)
trace22 = go.Scatter(x=[custom_labels[val] for val in New_Property_Filter['Unique_Values']], y=New_Property_Filter['AvgDV'], mode='lines', name="AvgDV", yaxis='y2', line=dict(color='red'))
layout = go.Layout(title=dict(text="Home Price by Age of Property", font=dict(size=16), y=0.99), xaxis=dict(title="Age of Property", tickfont=dict(size=12)), yaxis=dict(title="Number of Homes",tickfont=dict(size=12), range=[0, New_Property_Filter['Count'].max()], tick0=0, dtick=100000, showgrid=False), yaxis2=dict(title="Average Dependent Variable", tickfont=dict(size=12), overlaying='y', side='right', range=[0, New_Property_Filter['AvgDV'].max()], tick0=0, dtick=100000, showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=True, legend=dict(orientation='v', x=1, y=1.2))
Chart1_2 = go.Figure(data=[trace21, trace22], layout=layout)

#Duration_Relates_to_the_tenure
custom_labels = {"F": "Freehold", "L": "Leasehold "}
bar_colors = ["seagreen", "salmon"]
trace31 = go.Bar(x=[custom_labels[val] for val in Duration_Relates_to_the_tenure_Filter['Unique_Values']], y=Duration_Relates_to_the_tenure_Filter['Count'], name="Number of Homes", marker=dict(color=bar_colors), text=[count for count in Duration_Relates_to_the_tenure_Filter['Count']], showlegend=False)
trace32 = go.Scatter(x=[custom_labels[val] for val in Duration_Relates_to_the_tenure_Filter['Unique_Values']], y=Duration_Relates_to_the_tenure_Filter['AvgDV'], mode='lines', name="AvgDV", yaxis='y2', line=dict(color='red'))
layout = go.Layout(title=dict(text="Home Price by Property Ownership", font=dict(size=16), y=0.99), xaxis=dict(title="Property Ownership", tickfont=dict(size=12)), yaxis=dict(title="Number of Homes",tickfont=dict(size=12), range=[0, Duration_Relates_to_the_tenure_Filter['Count'].max()], tick0=0, dtick=100000, showgrid=False), yaxis2=dict(title="Average Dependent Variable", tickfont=dict(size=12), overlaying='y', side='right', range=[0, Duration_Relates_to_the_tenure_Filter['AvgDV'].max()], tick0=0, dtick=100000, showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=True, legend=dict(orientation='v', x=1, y=1.2))
Chart1_3 = go.Figure(data=[trace31, trace32], layout=layout)

#MonthYear
# custom_months = ['Jan2023','Feb2023','Mar2023','Apr2023','May2023','Jun2023','Jul2023']
# custom_labels = {"Jan2023": "Jan2023", "Feb2023": "Feb2023", "Mar2023": "Mar2023", "Apr2023": "Apr2023", "May2023": "May2023", "Jun2023": "Jun2023", "Jul2023": "Jul2023"}
bar_colors = ["royalblue", "yellowgreen", "crimson", "goldenrod", "deeppink", "rosybrown", "olivedrab"]
trace41 = go.Bar(x=MonthYear_Filter['Unique_Values'], y=MonthYear_Filter['Count'], name="Number of Homes", marker=dict(color=bar_colors), text=[count for count in MonthYear_Filter['Count']], showlegend=False)
trace42 = go.Scatter(x=MonthYear_Filter['Unique_Values'], y=MonthYear_Filter['AvgDV'], mode='lines', name="AvgDV", yaxis='y2', line=dict(color='red'))
layout = go.Layout(title=dict(text="Home Price by Time Period (In Months)", font=dict(size=16), y=0.99), xaxis=dict(title="Time Period (In Months)", tickfont=dict(size=12)), yaxis=dict(title="Number of Homes",tickfont=dict(size=12), range=[0, MonthYear_Filter['Count'].max()], tick0=0, dtick=100000, showgrid=False), yaxis2=dict(title="Average Dependent Variable", tickfont=dict(size=12), overlaying='y', side='right', range=[0, MonthYear_Filter['AvgDV'].max()], tick0=0, dtick=100000, showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=True, legend=dict(orientation='v', x=1, y=1.2))
Chart1_4 = go.Figure(data=[trace41, trace42], layout=layout)

left_column1, right_column1 = st.columns([2, 2])
with left_column1:
    st.plotly_chart(Chart1_1, use_container_width=True)
    st.plotly_chart(Chart1_2, use_container_width=True)
with right_column1:
    st.plotly_chart(Chart1_4, use_container_width=True)
    st.plotly_chart(Chart1_3, use_container_width=True)

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

# if selected_CorrSegment == 'Affordable':
#     Filter_bivardf = Filter_bivardf[Filter_bivardf['VariableB'].isin(Affordable_Varlist)]
# elif selected_CorrSegment == 'Mid_Range':
#     Filter_bivardf = Filter_bivardf[Filter_bivardf['VariableB'].isin(MidRange_Varlist)]
# elif selected_CorrSegment == 'Luxury':
#     Filter_bivardf = Filter_bivardf[Filter_bivardf['VariableB'].isin(Luxury_Varlist)]
# else:
#     Filter_bivardf

Filter_bivardf = Filter_bivardf.set_index('segment', drop=True)
Filter_bivardf = Filter_bivardf.sort_values(by='correlation_value', ascending=[False])
st.write(Filter_bivardf)


st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)

st.markdown("""""")
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Model Evaluation</h1>", unsafe_allow_html=True)
st.markdown("""""")

# Regression_Model_Metrics[['Model Name', 'Segment']] = Regression_Model_Metrics['Model Name'].str.split('_', 1, expand=True)
Regression_Model_Metrics[['Model Name', 'Segment']] = Regression_Model_Metrics['Model Name'].str.split('_', expand=True, n=1)

Segment_selected = st.selectbox('Please select Home Segment', Regression_Model_Metrics['Segment'].unique())
MevalMetric = Regression_Model_Metrics[Regression_Model_Metrics['Segment'] == Segment_selected]
MevalMetric['R2'] = MevalMetric['R2'].round(2)
MevalMetric['RMSE'] = MevalMetric['RMSE'].round(2)
MevalMetric['MSE'] = MevalMetric['MSE'].round(2)
MevalMetric['MAE'] = MevalMetric['MAE'].round(2)
# bar_colors = ["deepskyblue", "burlywood", "yellowgreen", "royalblue", "lightsalmon", "mediumturquoise", "coral", "goldenrod"]
# trace41 = go.Bar(x=MevalMetric['Model Name'], y=MevalMetric['R2'], name="R2", marker=dict(color=bar_colors))
# layout = go.Layout(title=dict(text="Model Evaluation - R2", font=dict(size=16)), xaxis=dict(title="Model Name", tickfont=dict(size=12)), yaxis=dict(title="R2", tickfont=dict(size=12), showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=False, legend=dict(orientation='v', x=1, y=1.2))
# Chart4_1 = go.Figure(data=[trace41], layout=layout)

bar_colors = ["deepskyblue", "burlywood", "yellowgreen", "royalblue", "lightsalmon", "mediumturquoise", "coral", "goldenrod"]
trace41 = go.Bar(x=MevalMetric['Model Name'], y=MevalMetric['R2'], name="Number of Homes", marker=dict(color=bar_colors), text=[count for count in MevalMetric['R2']], showlegend=False)
layout = go.Layout(title=dict(text="Model Evaluation - R2", font=dict(size=16)), xaxis=dict(title="Model Name", tickfont=dict(size=12)), yaxis=dict(title="R2", tickfont=dict(size=12), showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=False, legend=dict(orientation='v', x=1, y=1.2))
Chart4_1 = go.Figure(data=[trace41], layout=layout)

trace42 = go.Bar(x=MevalMetric['Model Name'], y=MevalMetric['RMSE'], name="Number of Homes", marker=dict(color=bar_colors), text=[count for count in MevalMetric['RMSE']], showlegend=False)
layout = go.Layout(title=dict(text="Model Evaluation - RMSE", font=dict(size=16)), xaxis=dict(title="Model Name", tickfont=dict(size=12)), yaxis=dict(title="RMSE", tickfont=dict(size=12), showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=False, legend=dict(orientation='v', x=1, y=1.2))
Chart4_2 = go.Figure(data=[trace42], layout=layout)

trace43 = go.Bar(x=MevalMetric['Model Name'], y=MevalMetric['MSE'], name="Number of Homes", marker=dict(color=bar_colors), text=[count for count in MevalMetric['MSE']], showlegend=False)
layout = go.Layout(title=dict(text="Model Evaluation - MSE", font=dict(size=16)), xaxis=dict(title="Model Name", tickfont=dict(size=12)), yaxis=dict(title="MSE", tickfont=dict(size=12), showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=False, legend=dict(orientation='v', x=1, y=1.2))
Chart4_3 = go.Figure(data=[trace43], layout=layout)

trace44 = go.Bar(x=MevalMetric['Model Name'], y=MevalMetric['MAE'], name="Number of Homes", marker=dict(color=bar_colors), text=[count for count in MevalMetric['MAE']], showlegend=False)
layout = go.Layout(title=dict(text="Model Evaluation - MAE", font=dict(size=16)), xaxis=dict(title="Model Name", tickfont=dict(size=12)), yaxis=dict(title="MAE", tickfont=dict(size=12), showgrid=False), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='grey', size=12), showlegend=False, legend=dict(orientation='v', x=1, y=1.2))
Chart4_4 = go.Figure(data=[trace44], layout=layout)

left_column1, right_column1 = st.columns([2, 2])
with left_column1:
    st.plotly_chart(Chart4_1, use_container_width=True)
    st.plotly_chart(Chart4_3, use_container_width=True)

with right_column1:
    st.plotly_chart(Chart4_2, use_container_width=True)
    st.plotly_chart(Chart4_4, use_container_width=True)

MevalMetric = MevalMetric.set_index('Model Name', drop=True)
st.write(MevalMetric)
st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)

#-----------------------------------------------------------------------------------------------------------------------
#Model Test Scored
st.markdown("""""")
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>Model Test Scores</h1>", unsafe_allow_html=True)
st.markdown("""""")

Choice = st.radio('Please select Home Segment', ['Affordable','MidRange','Luxury'])
if Choice == 'Affordable':
    dataset61 = Scored_Affordable.groupby('Property_Type')[['DVlog','y_pred_LR_AM','y_pred_DT_AM','y_pred_RF_AM','y_pred_XGB_AM','y_pred_NN_AM','y_pred_ANN_AM','y_pred_DNN_AM']].mean().reset_index()
    custom_labels = {"D": "Detached", "F": "Flats/Maisonettes", "T": "Terraced", "S": "Semi-Detached", "O": "Other"}
    trace61 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace62 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_LR_AM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace63 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_DT_AM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace64 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_RF_AM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace65 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_SVMR_AM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace66 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_XGB_AM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
    # trace67 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_NN_AM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
     #trace68 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_ANN_AM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace69 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_DNN_AM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Property Type", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Property Type"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart6_1 = go.Figure(data=[trace61, trace62, trace63, trace64, trace66, trace69], layout=layout)

    dataset71 = Scored_Affordable.groupby('New_Property')[['DVlog','y_pred_LR_AM','y_pred_DT_AM','y_pred_RF_AM','y_pred_XGB_AM','y_pred_NN_AM','y_pred_ANN_AM','y_pred_DNN_AM']].mean().reset_index()
    custom_labels = {"Y": "Newly Built Property", "N": "Established Residential Building"}
    trace71 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace72 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_LR_AM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace73 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_DT_AM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace74 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_RF_AM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace75 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_SVMR_AM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace76 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_XGB_AM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
     #trace77 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_NN_AM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
     #trace78 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_ANN_AM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace79 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_DNN_AM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Age of property", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Age of Property"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart7_1 = go.Figure(data=[trace71, trace72, trace73, trace74, trace76, trace79], layout=layout)

    dataset81 = Scored_Affordable.groupby('Duration_Relates_to_the_tenure')[['DVlog','y_pred_LR_AM','y_pred_DT_AM','y_pred_RF_AM','y_pred_XGB_AM','y_pred_NN_AM','y_pred_ANN_AM','y_pred_DNN_AM']].mean().reset_index()
    custom_labels = {"F": "Freehold", "L": "Leasehold "}
    trace81 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace82 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_LR_AM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace83 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_DT_AM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace84 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_RF_AM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace85 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_SVMR_AM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace86 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_XGB_AM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
     #trace87 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_NN_AM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
    # trace88 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_ANN_AM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace89 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_DNN_AM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Property Ownership", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Property Ownership"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart8_1 = go.Figure(data=[trace81, trace82, trace83, trace84, trace86, trace89], layout=layout)

    dataset91 = Scored_Affordable.groupby('MonthYear')[['DVlog','y_pred_LR_AM','y_pred_DT_AM','y_pred_RF_AM','y_pred_XGB_AM','y_pred_NN_AM','y_pred_ANN_AM','y_pred_DNN_AM']].mean().reset_index()
    trace91 = go.Line(x=dataset91['MonthYear'], y=dataset91['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace92 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_LR_AM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace93 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_DT_AM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace94 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_RF_AM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace95 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_SVMR_AM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace96 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_XGB_AM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
    # trace97 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_NN_AM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
     #trace98 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_ANN_AM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace99 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_DNN_AM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Time Period (In Months)", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Time Period (In Months)"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart9_1 = go.Figure(data=[trace91, trace92, trace93, trace94, trace96, trace99], layout=layout)

elif Choice == 'MidRange':
    dataset61 = Scored_MidRange.groupby('Property_Type')[['DVlog','y_pred_LR_MM','y_pred_DT_MM','y_pred_RF_MM','y_pred_XGB_MM','y_pred_NN_MM','y_pred_ANN_MM','y_pred_DNN_MM']].mean().reset_index()
    custom_labels = {"D": "Detached", "F": "Flats/Maisonettes", "T": "Terraced", "S": "Semi-Detached", "O": "Other"}
    trace61 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace62 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_LR_MM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace63 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_DT_MM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace64 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_RF_MM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace65 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_SVMR_MM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace66 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_XGB_MM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
     #trace67 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_NN_MM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
     #trace68 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_ANN_MM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace69 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_DNN_MM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Property Type", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Property Type"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart6_1 = go.Figure(data=[trace61, trace62, trace63, trace64, trace66, trace69], layout=layout)

    dataset71 = Scored_MidRange.groupby('New_Property')[['DVlog','y_pred_LR_MM','y_pred_DT_MM','y_pred_RF_MM','y_pred_XGB_MM','y_pred_NN_MM','y_pred_ANN_MM','y_pred_DNN_MM']].mean().reset_index()
    custom_labels = {"Y": "Newly Built Property", "N": "Established Residential Building"}
    trace71 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace72 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_LR_MM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace73 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_DT_MM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace74 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_RF_MM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace75 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_SVMR_MM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace76 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_XGB_MM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
    # trace77 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_NN_MM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
    # trace78 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_ANN_MM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace79 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_DNN_MM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Age of property", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Age of Property"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart7_1 = go.Figure(data=[trace71, trace72, trace73, trace74, trace76, trace79], layout=layout)

    dataset81 = Scored_MidRange.groupby('Duration_Relates_to_the_tenure')[['DVlog','y_pred_LR_MM','y_pred_DT_MM','y_pred_RF_MM','y_pred_XGB_MM','y_pred_NN_MM','y_pred_ANN_MM','y_pred_DNN_MM']].mean().reset_index()
    custom_labels = {"F": "Freehold", "L": "Leasehold "}
    trace81 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace82 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_LR_MM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace83 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_DT_MM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace84 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_RF_MM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace85 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_SVMR_MM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace86 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_XGB_MM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
    # trace87 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_NN_MM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
     #trace88 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_ANN_MM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace89 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_DNN_MM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Property Ownership", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Property Ownership"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart8_1 = go.Figure(data=[trace81, trace82, trace83, trace84, trace86, trace89], layout=layout)

    dataset91 = Scored_MidRange.groupby('MonthYear')[['DVlog','y_pred_LR_MM','y_pred_DT_MM','y_pred_RF_MM','y_pred_XGB_MM','y_pred_NN_MM','y_pred_ANN_MM','y_pred_DNN_MM']].mean().reset_index()
    trace91 = go.Line(x=dataset91['MonthYear'], y=dataset91['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace92 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_LR_MM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace93 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_DT_MM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace94 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_RF_MM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace95 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_SVMR_MM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace96 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_XGB_MM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
     #trace97 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_NN_MM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
     #trace98 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_ANN_MM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace99 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_DNN_MM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Time Period (In Months)", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Time Period (In Months)"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart9_1 = go.Figure(data=[trace91, trace92, trace93, trace94, trace96, trace99], layout=layout)
elif Choice == 'Luxury':
    dataset61 = Scored_Luxury.groupby('Property_Type')[['DVlog','y_pred_LR_LM','y_pred_DT_LM','y_pred_RF_LM','y_pred_XGB_LM','y_pred_NN_LM','y_pred_ANN_LM','y_pred_DNN_LM']].mean().reset_index()
    custom_labels = {"D": "Detached", "F": "Flats/Maisonettes", "T": "Terraced", "S": "Semi-Detached", "O": "Other"}
    trace61 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace62 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_LR_LM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace63 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_DT_LM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace64 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_RF_LM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace65 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_SVMR_LM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace66 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_XGB_LM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
    # trace67 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_NN_LM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
    # trace68 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_ANN_LM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace69 = go.Line(x=[custom_labels[val] for val in dataset61['Property_Type']], y=dataset61['y_pred_DNN_LM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Property Type", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Property Type"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart6_1 = go.Figure(data=[trace61, trace62, trace63, trace64, trace66, trace69], layout=layout)

    dataset71 = Scored_Luxury.groupby('New_Property')[['DVlog','y_pred_LR_LM','y_pred_DT_LM','y_pred_RF_LM','y_pred_XGB_LM','y_pred_NN_LM','y_pred_ANN_LM','y_pred_DNN_LM']].mean().reset_index()
    custom_labels = {"Y": "Newly Built Property", "N": "Established Residential Building"}
    trace71 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace72 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_LR_LM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace73 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_DT_LM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace74 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_RF_LM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace75 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_SVMR_LM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace76 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_XGB_LM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
    # trace77 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_NN_LM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
     #trace78 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_ANN_LM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace79 = go.Line(x=[custom_labels[val] for val in dataset71['New_Property']], y=dataset71['y_pred_DNN_LM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Age of property", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Age of Property"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart7_1 = go.Figure(data=[trace71, trace72, trace73, trace74, trace76, trace79], layout=layout)

    dataset81 = Scored_Luxury.groupby('Duration_Relates_to_the_tenure')[['DVlog','y_pred_LR_LM','y_pred_DT_LM','y_pred_RF_LM','y_pred_XGB_LM','y_pred_NN_LM','y_pred_ANN_LM','y_pred_DNN_LM']].mean().reset_index()
    custom_labels = {"F": "Freehold", "L": "Leasehold "}
    trace81 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace82 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_LR_LM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace83 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_DT_LM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace84 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_RF_LM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace85 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_SVMR_LM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace86 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_XGB_LM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
    # trace87 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_NN_LM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
     #trace88 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_ANN_LM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace89 = go.Line(x=[custom_labels[val] for val in dataset81['Duration_Relates_to_the_tenure']], y=dataset81['y_pred_DNN_LM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Property Ownership", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title="Property Ownership"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart8_1 = go.Figure(data=[trace81, trace82, trace83, trace84, trace86, trace89], layout=layout)

    dataset91 = Scored_Luxury.groupby('MonthYear')[['DVlog','y_pred_LR_LM','y_pred_DT_LM','y_pred_RF_LM','y_pred_XGB_LM','y_pred_NN_LM','y_pred_ANN_LM','y_pred_DNN_LM']].mean().reset_index()
    trace91 = go.Line(x=dataset91['MonthYear'], y=dataset91['DVlog'], name="Actual Price", mode='lines', marker=dict(color='red'), showlegend=True)
    trace92 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_LR_LM'], name="Linear Regression", mode='lines', marker=dict(color='deepskyblue'), showlegend=True)
    trace93 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_DT_LM'], name="Decision Tree", mode='lines',marker=dict(color='burlywood'), showlegend=True)
    trace94 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_RF_LM'], name='Random Forest', mode='lines',marker=dict(color='blue'), showlegend=True)
    # trace95 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_SVMR_LM'], name='SVM', mode='lines',marker=dict(color='royalblue'), showlegend=True)
    trace96 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_XGB_LM'], name='XGBoost', mode='lines',marker=dict(color='lightsalmon'), showlegend=True)
    # trace97 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_NN_LM'], name='Neural Networks', mode='lines',marker=dict(color='mediumturquoise'), showlegend=True)
     #trace98 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_ANN_LM'], name='Artificial Neural Networks', mode='lines',marker=dict(color='coral'), showlegend=True)
    trace99 = go.Line(x=dataset91['MonthYear'], y=dataset91['y_pred_DNN_LM'], name='Deep Neural Networks', mode='lines',marker=dict(color='goldenrod'), showlegend=True)
    layout = go.Layout(title=dict(text="Testset - Actual vs Predicted by Time Period (In Months)", font=dict(size=16)),plot_bgcolor='white', paper_bgcolor='white',xaxis=dict(title=" Time Period (In Months)"), yaxis=dict(title="Actual vs Predicted",titlefont=dict(size=14)))
    Chart9_1 = go.Figure(data=[trace91, trace92, trace93, trace94, trace96, trace99], layout=layout)
else:
    st.markdown("No selection or no data present")

left_column1, right_column1 = st.columns([2, 2])
with left_column1:
    st.plotly_chart(Chart6_1, use_container_width=True)
    st.plotly_chart(Chart7_1, use_container_width=True)
with right_column1:
    st.plotly_chart(Chart9_1, use_container_width=True)
    st.plotly_chart(Chart8_1, use_container_width=True)

st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 16px; margin: 0; padding: 0;'>End of Report</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 5px 0; padding: 0;'>", unsafe_allow_html=True)
