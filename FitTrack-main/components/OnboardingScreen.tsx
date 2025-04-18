import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { COLORS, SIZES, SHADOWS } from '../constants/theme';

export interface OnboardingData {
  height: string;
  weight: string;
  heightUnit: 'cm' | 'in';
  weightUnit: 'kg' | 'lbs';
  bodyFat: string;
  gender: 'male' | 'female';
  activityLevel: string;
  fitnessGoal: string;
  weeklyWorkouts: string;
  wakeTime: string;
  workoutTime: string;
  sleepTime: string;
  macroSplit: {
    protein: number;
    carbs: number;
    fats: number;
  };
  mealCount: string;
}

interface OnboardingScreenProps {
  onComplete: (data: OnboardingData) => void;
}

const ACTIVITY_LEVELS = [
  'Sedentary (little or no exercise)',
  'Lightly active (light exercise/sports 1-3 days/week)',
  'Moderately active (moderate exercise/sports 3-5 days/week)',
  'Very active (hard exercise/sports 6-7 days/week)',
  'Extra active (very hard exercise/sports & physical job or training twice per day)',
];

const FITNESS_GOALS = [
  'Lose Weight',
  'Build Muscle',
  'Improve Strength',
  'Increase Endurance',
  'Maintain Fitness',
];

const OnboardingScreen = ({ onComplete }: OnboardingScreenProps) => {
  const [step, setStep] = useState(1);
  const [data, setData] = useState<OnboardingData>({
    height: '',
    weight: '',
    heightUnit: 'in',
    weightUnit: 'lbs',
    bodyFat: '',
    gender: 'male',
    activityLevel: '',
    fitnessGoal: '',
    weeklyWorkouts: '',
    wakeTime: '06:00',
    workoutTime: '17:00',
    sleepTime: '22:00',
    macroSplit: {
      protein: 30,
      carbs: 35,
      fats: 35
    },
    mealCount: '3'
  });

  const handleNext = () => {
    if (step === 1 && (!data.height || !data.weight)) {
      Alert.alert('Error', 'Please fill in height and weight');
      return;
    }
    if (step === 2 && !data.activityLevel) {
      Alert.alert('Error', 'Please select your activity level');
      return;
    }
    if (step === 3 && (!data.fitnessGoal || !data.weeklyWorkouts)) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }
    if (step === 4 && (!data.wakeTime || !data.workoutTime || !data.sleepTime)) {
      Alert.alert('Error', 'Please set all time preferences');
      return;
    }
    if (step < 4) {
      setStep(step + 1);
    } else {
      onComplete(data);
    }
  };

  const renderStep1 = () => (
    <View style={styles.stepContainer}>
      <Text style={styles.title}>Your Current Stats</Text>
      <Text style={styles.subtitle}>Let's get your basic measurements</Text>
      
      <View style={styles.inputContainer}>
        <Text style={styles.label}>Gender</Text>
        <View style={styles.genderContainer}>
          <TouchableOpacity
            style={[
              styles.genderButton,
              data.gender === 'male' && styles.selectedOption,
            ]}
            onPress={() => setData({ ...data, gender: 'male' })}
          >
            <Text style={[
              styles.genderButtonText,
              data.gender === 'male' && styles.selectedOptionText,
            ]}>Male</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.genderButton,
              data.gender === 'female' && styles.selectedOption,
            ]}
            onPress={() => setData({ ...data, gender: 'female' })}
          >
            <Text style={[
              styles.genderButtonText,
              data.gender === 'female' && styles.selectedOptionText,
            ]}>Female</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Height</Text>
        <View style={styles.unitContainer}>
          <TextInput
            style={[styles.input, { flex: 1 }]}
            value={data.height}
            onChangeText={(text) => setData({ ...data, height: text })}
            keyboardType="numeric"
            placeholder={data.heightUnit === 'in' ? "71" : "180"}
          />
          <TouchableOpacity
            style={styles.unitButton}
            onPress={() => setData({ ...data, heightUnit: data.heightUnit === 'in' ? 'cm' : 'in' })}
          >
            <Text style={styles.unitButtonText}>{data.heightUnit}</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Weight</Text>
        <View style={styles.unitContainer}>
          <TextInput
            style={[styles.input, { flex: 1 }]}
            value={data.weight}
            onChangeText={(text) => setData({ ...data, weight: text })}
            keyboardType="numeric"
            placeholder={data.weightUnit === 'lbs' ? "180" : "82"}
          />
          <TouchableOpacity
            style={styles.unitButton}
            onPress={() => setData({ ...data, weightUnit: data.weightUnit === 'lbs' ? 'kg' : 'lbs' })}
          >
            <Text style={styles.unitButtonText}>{data.weightUnit}</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Body Fat % (optional)</Text>
        <TextInput
          style={styles.input}
          value={data.bodyFat}
          onChangeText={(text) => setData({ ...data, bodyFat: text })}
          keyboardType="numeric"
          placeholder="Enter your body fat percentage"
        />
      </View>
    </View>
  );

  const renderStep2 = () => (
    <View style={styles.stepContainer}>
      <Text style={styles.title}>Activity Level</Text>
      <Text style={styles.subtitle}>How active are you on a typical week?</Text>
      
      {ACTIVITY_LEVELS.map((level, index) => (
        <TouchableOpacity
          key={index}
          style={[
            styles.optionButton,
            data.activityLevel === level && styles.selectedOption,
          ]}
          onPress={() => setData({ ...data, activityLevel: level })}
        >
          <Text style={[
            styles.optionText,
            data.activityLevel === level && styles.selectedOptionText,
          ]}>
            {level}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderStep3 = () => (
    <View style={styles.stepContainer}>
      <Text style={styles.title}>Your Goals</Text>
      <Text style={styles.subtitle}>Let's set your fitness objectives</Text>
      
      <Text style={styles.label}>What's your primary goal?</Text>
      {FITNESS_GOALS.map((goal, index) => (
        <TouchableOpacity
          key={index}
          style={[
            styles.optionButton,
            data.fitnessGoal === goal && styles.selectedOption,
          ]}
          onPress={() => setData({ ...data, fitnessGoal: goal })}
        >
          <Text style={[
            styles.optionText,
            data.fitnessGoal === goal && styles.selectedOptionText,
          ]}>
            {goal}
          </Text>
        </TouchableOpacity>
      ))}

      <View style={styles.inputContainer}>
        <Text style={styles.label}>How many workouts per week?</Text>
        <TextInput
          style={styles.input}
          value={data.weeklyWorkouts}
          onChangeText={(text) => setData({ ...data, weeklyWorkouts: text })}
          keyboardType="numeric"
          placeholder="Enter number of workouts"
        />
      </View>
    </View>
  );

  const renderStep4 = () => (
    <View style={styles.stepContainer}>
      <Text style={styles.title}>Daily Schedule</Text>
      <Text style={styles.subtitle}>Let's optimize your meal timing</Text>
      
      <View style={styles.inputContainer}>
        <Text style={styles.label}>Wake Up Time</Text>
        <TextInput
          style={styles.input}
          value={data.wakeTime}
          onChangeText={(text) => setData({ ...data, wakeTime: text })}
          placeholder="06:00"
          keyboardType="numbers-and-punctuation"
        />
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Workout Time</Text>
        <TextInput
          style={styles.input}
          value={data.workoutTime}
          onChangeText={(text) => setData({ ...data, workoutTime: text })}
          placeholder="17:00"
          keyboardType="numbers-and-punctuation"
        />
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Sleep Time</Text>
        <TextInput
          style={styles.input}
          value={data.sleepTime}
          onChangeText={(text) => setData({ ...data, sleepTime: text })}
          placeholder="22:00"
          keyboardType="numbers-and-punctuation"
        />
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Macro Split (%)</Text>
        <View style={styles.macroContainer}>
          <View style={styles.macroInput}>
            <Text style={styles.macroLabel}>Protein</Text>
            <TextInput
              style={[styles.input, styles.macroValue]}
              value={data.macroSplit.protein.toString()}
              onChangeText={(text) => setData({
                ...data,
                macroSplit: {
                  ...data.macroSplit,
                  protein: parseInt(text) || 0
                }
              })}
              keyboardType="numeric"
              placeholder="30"
            />
          </View>
          <View style={styles.macroInput}>
            <Text style={styles.macroLabel}>Carbs</Text>
            <TextInput
              style={[styles.input, styles.macroValue]}
              value={data.macroSplit.carbs.toString()}
              onChangeText={(text) => setData({
                ...data,
                macroSplit: {
                  ...data.macroSplit,
                  carbs: parseInt(text) || 0
                }
              })}
              keyboardType="numeric"
              placeholder="35"
            />
          </View>
          <View style={styles.macroInput}>
            <Text style={styles.macroLabel}>Fats</Text>
            <TextInput
              style={[styles.input, styles.macroValue]}
              value={data.macroSplit.fats.toString()}
              onChangeText={(text) => setData({
                ...data,
                macroSplit: {
                  ...data.macroSplit,
                  fats: parseInt(text) || 0
                }
              })}
              keyboardType="numeric"
              placeholder="35"
            />
          </View>
        </View>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>How many meals per day?</Text>
        <Picker
          selectedValue={data.mealCount}
          onValueChange={(value: string) => setData({ ...data, mealCount: value })}
          style={styles.picker}
        >
          <Picker.Item label="3 meals" value="3" />
          <Picker.Item label="4 meals" value="4" />
          <Picker.Item label="5 meals" value="5" />
        </Picker>
      </View>
    </View>
  );

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.progressBar}>
          <View style={[styles.progressFill, { width: `${(step / 4) * 100}%` }]} />
        </View>

        {step === 1 && renderStep1()}
        {step === 2 && renderStep2()}
        {step === 3 && renderStep3()}
        {step === 4 && renderStep4()}

        <TouchableOpacity style={styles.nextButton} onPress={handleNext}>
          <Text style={styles.nextButtonText}>
            {step === 4 ? 'Complete' : 'Next'}
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scrollContent: {
    flexGrow: 1,
    padding: 20,
  },
  progressBar: {
    height: 4,
    backgroundColor: COLORS.border,
    borderRadius: 2,
    marginBottom: 30,
  },
  progressFill: {
    height: '100%',
    backgroundColor: COLORS.primary,
    borderRadius: 2,
  },
  stepContainer: {
    flex: 1,
  },
  title: {
    fontSize: SIZES.xLarge,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 10,
  },
  subtitle: {
    fontSize: SIZES.medium,
    color: COLORS.subtext,
    marginBottom: 30,
  },
  inputContainer: {
    marginBottom: 20,
  },
  label: {
    fontSize: SIZES.medium,
    color: COLORS.text,
    marginBottom: 8,
  },
  input: {
    backgroundColor: COLORS.card,
    borderRadius: 10,
    padding: 15,
    fontSize: SIZES.medium,
    ...SHADOWS.small,
  },
  optionButton: {
    backgroundColor: COLORS.card,
    borderRadius: 10,
    padding: 15,
    marginBottom: 10,
    ...SHADOWS.small,
  },
  selectedOption: {
    backgroundColor: COLORS.primary,
  },
  optionText: {
    fontSize: SIZES.medium,
    color: COLORS.text,
  },
  selectedOptionText: {
    color: COLORS.card,
  },
  nextButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 10,
    padding: 15,
    alignItems: 'center',
    marginTop: 20,
    ...SHADOWS.small,
  },
  nextButtonText: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  unitContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  unitButton: {
    backgroundColor: COLORS.accent,
    padding: 15,
    borderRadius: 10,
    minWidth: 60,
    alignItems: 'center',
  },
  unitButtonText: {
    color: COLORS.card,
    fontWeight: 'bold',
    fontSize: SIZES.medium,
  },
  genderContainer: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 20,
  },
  genderButton: {
    flex: 1,
    backgroundColor: COLORS.card,
    borderRadius: 10,
    padding: 15,
    alignItems: 'center',
    ...SHADOWS.small,
  },
  genderButtonText: {
    fontSize: SIZES.medium,
    color: COLORS.text,
  },
  macroContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
  },
  macroInput: {
    flex: 1,
  },
  macroLabel: {
    fontSize: SIZES.small,
    color: COLORS.text,
    marginBottom: 4,
  },
  macroValue: {
    textAlign: 'center',
  },
  picker: {
    backgroundColor: COLORS.card,
    borderRadius: 10,
    padding: 15,
    ...SHADOWS.small,
  },
});

export default OnboardingScreen; 