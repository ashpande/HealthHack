import React, {useState} from 'react';
import { Text } from 'react-native';
import MapView from 'react-native-maps';
import {Marker} from 'react-native-maps';

const Map = ({latitude, longitude, title, desc}) => {
  const [region, setRegion] = useState({
    latitude: 12.93539,
    longitude: 77.534851,
    latitudeDelta: 0.0922,
    longitudeDelta: 0.0421,
  });
  const [marker, setMarker] = useState({title, desc})

  return (
    <MapView
      region={region}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        borderRadius: 10,
      }}
    >
      <Marker
        coordinate={{latitude: 12.93539, longitude: 77.534851}}
        title={marker.title}
        description={marker.desc}
      />
    </MapView>  
    ); 
}

export default Map;