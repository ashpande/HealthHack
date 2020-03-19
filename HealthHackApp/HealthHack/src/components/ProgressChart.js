import React from 'react';
import { ProgressCircle }  from 'react-native-svg-charts';

const ProgressChart = ({height, progress, color}) => {
  console.log(`rgba(${color.split('(')[1].split(')')[0]},0.8)`);
  return (
    <ProgressCircle
      style={ { height } }
      progress={ progress }
      progressColor={ color }
      strokeWidth={ 7 }
      backgroundColor={ color }
    />
  );
}

export default ProgressChart;