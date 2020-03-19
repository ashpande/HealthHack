import { createAppContainer } from 'react-navigation';
import { createStackNavigator } from 'react-navigation-stack';
import { createBottomTabNavigator } from 'react-navigation-tabs';
import HomeScreen from './src/screens/HomeScreen';
import MonitorMainScreen from './src/screens/MonitorMainScreen';

const navigator = createStackNavigator({
  Home: HomeScreen,
  MonitorMain: MonitorMainScreen,
},
{
  initialRouteName: 'Home',
  navigationOptions: ({navigation}) => ({
      header: <AppBar title={navigation.getParam('appBar', {title: ''}).title}/>
  }),
  defaultNavigationOptions: {
    title: 'Health Hack'
  }
});

export default createAppContainer(navigator);