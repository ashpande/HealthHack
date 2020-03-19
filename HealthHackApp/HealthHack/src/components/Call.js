	
import React, { Component } from 'react';
 
import { StyleSheet, Text, View, TouchableOpacity, Linking, Platform } from 'react-native';

export default class mainapp extends Component {
 
  dialCall = () => {
 
    let phoneNumber = '';
 
    if (Platform.OS === 'android') {
      phoneNumber = 'tel:108';
    }
    else {
      phoneNumber = 'telprompt:108';
    }
 
    Linking.openURL(phoneNumber);
  };
 
  render() {
    return (
      <View style={styles.MainContainer}>
 
        <TouchableOpacity onPress={this.dialCall} activeOpacity={0.7} style={styles.button} >
 
          <Text style={styles.TextStyle}>Call Ambulance</Text>
 
        </TouchableOpacity>
 
      </View>
 
    );
  }
}
 
const styles = StyleSheet.create({
 
  MainContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
  },
  button: {
 
    width: '60%',
    padding: 15,
    backgroundColor: 'rgba(255,0,0,0.8)',
    borderRadius: 10,
  },
 
  TextStyle: {
    color: '#fff',
    fontSize: 18,
    textAlign: 'center',
  }
 
});