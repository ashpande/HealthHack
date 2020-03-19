import React, {useState} from 'react';
import { Text, View, StyleSheet, TouchableOpacity, Image, ScrollView, FlatList } from 'react-native';
import { black } from 'ansi-colors';
import ProgressChart from '../components/ProgressChart';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import LineGraph from '../components/LineGraph';

const HomeScreen = ({navigation}) => {
  const  [bodySmallValues, setBodySmallValues] = useState({
    temp: 26,
    bac: 0.002,
    pulse: 58
  });

  return (
    <ScrollView style={{flex:1}}>
      <View style={styles.startMonitoringView}>
        <TouchableOpacity 
          style={styles.startMonitoringButton}
          onPress={() => navigation.navigate('MonitorMain')}
        >
          <Text style={{color: 'white'}}>Start Monitoring</Text>
        </TouchableOpacity>
      </View>
      <View style={styles.previousMonitoringView}>
        <Text style={styles.previousMonitoringHeading}>Most Recent Session</Text>
        <ScrollView style={{flex:1}}>
          <View style={styles.bodySmallValuesView}>
            <View style={styles.bodySmallValues}>
              <ProgressChart height={120} progress={10} color="rgb(0,150,136)" />
              <Text style={styles.bodySmallValuesTextContent}>
                {bodySmallValues.temp}
                <MaterialCommunityIcons name="temperature-celsius" size={22}/>
              </Text>
            </View>
            <View style={styles.bodySmallValues}>
              <ProgressChart height={120} progress={10} color="rgb(254,192,7)" />
              <Text style={styles.bodySmallValuesTextContent}>
                {bodySmallValues.bac}<MaterialCommunityIcons name="water-percent" size={25}/>
              </Text>
            </View>
            <View style={styles.bodySmallValues}>
              <ProgressChart height={120} progress={10} color="rgb(254,87,34)" />
              <Text style={styles.bodySmallValuesTextContent}>
                {bodySmallValues.pulse}/min
              </Text>
            </View>
          </View>
          <View style={styles.bodyBigValuesView}>
            <View style={styles.bodyBigValuesGraph}>
              <LineGraph lineStyles={styles.lineGraph} textContent={"EMG Graph"} />
              <LineGraph lineStyles={styles.lineGraph} textContent={"ECG Graph"} />
            </View>
          </View>
        </ScrollView>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  startMonitoringView:{
    height: 300,
    alignItems: 'center',
    justifyContent: 'center',
  },
  previousMonitoringView: {
    flex: 1,
    marginHorizontal: 15,
  },
  startMonitoringButton: {
    borderRadius:5,
    backgroundColor: '#2196F3',
    padding: 15
  },
  previousMonitoringHeading: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15
  },
  previousMonitoringSubHeading:{
    fontSize: 14,
    fontWeight: "200",
    color: '#5f9ea0'
  },
  bodySmallValuesView:{
    flexDirection: 'row',
    justifyContent: 'center',
    alignContent: 'center'
  },
  bodySmallValues: {
    flex: 1,
  },
  bodySmallValuesTextContent: {
    position: "absolute", 
    alignSelf: 'center',
    top: 42,
    fontSize: 25,
    fontWeight: "600"
  },
  bodyBigValuesView:{
    
  },
  bodyBigValuesGraph:{
    flex: 1, 
    alignItems: 'center',
    marginBottom: 35
  },
  lineGraph: { 
    borderRadius: 5, 
    marginTop: 35
  }
});

export default HomeScreen;