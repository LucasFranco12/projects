import React from 'react';
import { Image, View, StyleSheet } from 'react-native';

interface LogoImageProps {
  size?: number;
  style?: object;
  withText?: boolean;
}

const LogoImage = ({ size = 100, style = {}, withText = false }: LogoImageProps) => {
  const logoSource = withText 
    ? require('../assets/fitracklogo_withText.png')  // Update this filename to match your actual filename
    : require('../assets/fitracklogo_withnoText.png');           // Update this filename to match your actual filename
  
  return (
    <View style={[styles.container, { width: size, height: size }, style]}>
      <Image 
        source={logoSource}
        style={styles.image}
        resizeMode="contain"
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: '100%',
    height: '100%',
  },
});

export default LogoImage; 