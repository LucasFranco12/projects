import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES } from '../constants/theme';

interface DateSelectorProps {
  selectedDate: Date;
  onDateChange: (date: Date) => void;
}

const DateSelector = ({ selectedDate, onDateChange }: DateSelectorProps) => {
  const isToday = (date: Date) => {
    const today = new Date();
    return (
      date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear()
    );
  };

  const formatDisplayDate = (date: Date) => {
    const options: Intl.DateTimeFormatOptions = {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    };
    return date.toLocaleDateString('en-US', options);
  };

  const goToPreviousDay = () => {
    const newDate = new Date(selectedDate);
    newDate.setDate(newDate.getDate() - 1);
    onDateChange(newDate);
  };

  const goToNextDay = () => {
    const newDate = new Date(selectedDate);
    newDate.setDate(newDate.getDate() + 1);
    onDateChange(newDate);
  };

  const goToToday = () => {
    onDateChange(new Date());
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={goToPreviousDay} style={styles.dateButton}>
        <Ionicons name="chevron-back" size={24} color={COLORS.text} />
      </TouchableOpacity>
      
      <TouchableOpacity onPress={goToToday} style={styles.dateContainer}>
        <Text style={styles.dateText}>{formatDisplayDate(selectedDate)}</Text>
        {!isToday(selectedDate) && (
          <Text style={styles.todayHint}>(Tap to go to today)</Text>
        )}
      </TouchableOpacity>
      
      <TouchableOpacity onPress={goToNextDay} style={styles.dateButton}>
        <Ionicons name="chevron-forward" size={24} color={COLORS.text} />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
  },
  dateButton: {
    padding: 8,
  },
  dateContainer: {
    alignItems: 'center',
    paddingHorizontal: 16,
  },
  dateText: {
    fontSize: SIZES.medium,
    color: COLORS.text,
    fontWeight: 'bold',
  },
  todayHint: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
    marginTop: 4,
  },
});

export default DateSelector; 