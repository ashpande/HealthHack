import React from 'react';
import {LineChart} from 'react-native-chart-kit';
import { Line } from 'react-native-svg';
import { Dimensions, Text, StyleSheet } from "react-native";
const screenWidth = Dimensions.get("window").width;

const data_bak = {
  datasets: [{
    data: [ 20, 45, 28, 80, 99, 43 ],
    color: (opacity = 1) => `rgba(134, 65, 244, ${opacity})`,
    strokeWidth: 2
  }]
}
const chartConfig = {
  backgroundGradientFrom: '#1E2923',
  backgroundGradientFromOpacity: 0.05,
  backgroundGradientTo: '#1E2923',
  backgroundGradientToOpacity: 0.05,
  color: (opacity = 1) => `rgba(0, 0, 52, ${opacity})`,
  strokeWidth: 3, 
  barPercentage:0.8
}
const LineGraph = ({lineStyles, textContent, data}) => {
  const final_data = data? {datasets: [{
    data,
    color: (opacity = 1) => `rgba(134, 65, 244, ${opacity})`,
    strokeWidth: 2
  }]} : data_bak;
  // console.log(final_data);
  return (
    <>
      <LineChart
        style={lineStyles}
        data={final_data}
        width={screenWidth - 30}
        height={220}
        chartConfig={chartConfig}
      />
      <Text style={styles.textContent}>{textContent}</Text>
    </>
  );
}

const styles = StyleSheet.create({
  textContent: {
    fontSize: 24,
    fontWeight: "600",
    top: -100,
    marginLeft: 100,
    color: 'rgba(0,0,0,0.5)'
  }
});

export default LineGraph;