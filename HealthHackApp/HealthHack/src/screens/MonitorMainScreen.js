import React, {useState} from 'react';
import {Text, View, StyleSheet, Image, ScrollView, Button} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import LineGraph from '../components/LineGraph';
import { FlatList } from 'react-native-gesture-handler';
import Map from '../components/Map';
import Call from '../components/Call';
import useInterval from '../hooks/UseInterval'

const MonitorMainScreen = ({navigation}) => {
  const emg_json = require('../../assets/csvjson.json');
  const emg_json_array = emg_json.map(x=>x.emg);
  const ecg_json = require('../../assets/ecg.json');
  const ecg_json_array = ecg_json.map(x=>x.ecg);
  const [emg, setEmg] = useState([emg_json_array[0]]);
  const [ecg, setEcg] = useState([ecg_json_array[0]]);
  const [count, setCount] = useState(1);

  useInterval(()=>{
    // if(isNan(emg_json_array[count])){
    //   return;
    // }
    if(emg.length==20){
      setEmg([...emg.slice(1), emg_json_array[count]]);
      setEcg([...ecg.slice(1), ecg_json_array[count]]);
    }else{
      setEmg([...emg, emg_json_array[count]]);
      setEcg([...ecg, ecg_json_array[count]]);
    }
    setCount(count+1);
  }, 2000)
  
  // setEmg(emg.slice(0, 50).map(x=>x.emg));

  // json.forEach((item)=>{
  //   setEmg([...emg, item]);
  // })
  
  const  [smallValues, setSmallValues] = useState({
    temp: 37.1,
    bac: 0.002,
    pulse: 59
  });

  const  [locationValues, setLocationValues] = useState({
    temp: 25.70,
    pressure: 91991,
    alt: 806.74,
    gps: {
      latitude: 12.93539,
      longitude: 77.534851,
      title: "PES University",
      desc: "PES University Hackathon HashCode 2019"
    }
  });

  const thermalImagesBW = [
    require('../../assets/thermal-images/1.png'),
    require('../../assets/thermal-images/2.png'),
    require('../../assets/thermal-images/3.png'),
    require('../../assets/thermal-images/4.png'),
  ];

  const thermalImagesColor = [
    require('../../assets/thermal-images/6.png'),
    require('../../assets/thermal-images/7.png'),
    require('../../assets/thermal-images/8.png'),
    require('../../assets/thermal-images/9.png'),
  ];
  
  return (
    <ScrollView style={{flex: 1}}>
      {/* <Button 
        title="Goto PDF Page"
        onPress={()=>navigation.navigate("Report")}
      /> */}
      <View style={styles.section}>
        <Text style={styles.heading}>Patient</Text>
        <LineGraph 
          data={emg}
          lineStyles={styles.lineGraph} 
          textContent={"EMG Graph"}
        />
        <LineGraph 
          data={ecg}
          lineStyles={styles.lineGraph} 
          textContent={"ECG Graph"}
        />
        <View>
          <View style={{...styles.boxes, backgroundColor: '#F5F5F5'}}>
            <Text style={styles.boxesChild} >Temperature</Text>
            <Text style={{...styles.boxesChild, left: 60}} >
              {smallValues.temp}
              <MaterialCommunityIcons name="temperature-celsius" size={15}/>
            </Text>
          </View>
          <View style={{...styles.boxes, backgroundColor: '#F5F5F5'}}>
            <Text style={styles.boxesChild} >Blood Alcohol Conc.</Text>
            <Text style={{...styles.boxesChild, left: 60}} >
              {smallValues.bac}
              <MaterialCommunityIcons name="percent" /><Text style={{fontSize: 14}}>vol</Text>
            </Text>
          </View>
          <View style={{...styles.boxes, backgroundColor: '#F5F5F5'}}>
            <Text style={styles.boxesChild} >Pulse Rate</Text>
            <Text style={{...styles.boxesChild, left: 60}} >
              {smallValues.pulse}<Text style={{fontSize: 14}}>/min</Text>
            </Text>
          </View>
        </View>
        <View>
          <Text style={styles.imagesText}>Thermal Images</Text>
          <FlatList 
            horizontal
            showsHorizontalScrollIndicator={false}
            data={thermalImagesBW}
            keyExtractor={(item)=>item.toString()}
            renderItem={({item}) => {
              return <Image source={item} style={styles.imageStyle} />                
            }}
          />
          <FlatList 
            horizontal
            showsHorizontalScrollIndicator={false}
            data={thermalImagesColor}
            keyExtractor={(item)=>item.toString()}
            renderItem={({item}) => {
              return <Image source={item} style={styles.imageStyle} />                
            }}
          />
        </View>
      </View>
      <View style={styles.section}>
        <Text style={styles.heading}>Location</Text>
        <View style={{...styles.boxes, backgroundColor: '#F5F5F5'}}>
          <Text style={styles.boxesChild} >Temperature</Text>
          <Text style={{...styles.boxesChild, left: 60}} >
            {smallValues.temp}
            <MaterialCommunityIcons name="temperature-celsius" size={15}/>
          </Text>
        </View>
        <View style={{...styles.boxes, backgroundColor: '#F5F5F5'}}>
          <Text style={styles.boxesChild} >Pressure</Text>
          <Text style={{...styles.boxesChild, left: 60}} >
            {locationValues.pressure}
            <Text style={{fontSize: 14}}>Pa</Text>
          </Text>
        </View>
        <View style={{...styles.boxes, backgroundColor: '#F5F5F5'}}>
          <Text style={styles.boxesChild} >Altitude</Text>
          <Text style={{...styles.boxesChild, left: 60}} >
            {locationValues.alt}m
            <Text style={{fontSize: 12}}>above sea level</Text>
          </Text>
        </View>
        <View style={{height: 300, marginVertical: 15, borderRadius: 10}}>
          <Map 
            latitude={locationValues.gps.lat} 
            longitude={locationValues.gps.lon}
            title={locationValues.gps.title}
            desc={locationValues.gps.desc}
          />
        </View>
      </View>
      <Call />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  heading: {
    fontSize:20,
    fontWeight:'600',
    marginVertical: 10,
  }, 
  section: {
    marginHorizontal: 15
  },
  lineGraph: { 
    borderRadius: 10, 
    alignSelf: 'center'
  },
  boxes: {
    flexDirection: 'row',
    paddingVertical: 25,
    paddingHorizontal: 15,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 10,
    marginTop: 15
  },
  boxesChild: {
    flex: 1,
    fontSize: 21,
    // // textAlign: 'center'
    // borderWidth: 2,
    // borderColor: 'black'
  },
  imagesText:{
    fontSize: 20,
    fontWeight: "500",
    marginTop: 15,
    marginBottom: 5
  },
  imageStyle: {
    height: 100,
    width: 100,
    marginRight: 10,
    marginBottom: 10,
    borderRadius: 10
  }
});

export default MonitorMainScreen;