import React, { useState } from 'react';
import { StyleSheet, Text, View, ScrollView, Image, TextInput, TouchableOpacity, Animated } from 'react-native';
import { COLORS, SIZES, SHADOWS } from '../constants/theme';
import { UserData } from '../App';
import LogoImage from '../components/LogoImage';
import { Ionicons } from '@expo/vector-icons';

interface HomeScreenProps {
  userData: UserData;
}

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

const HomeScreen = ({ userData }: HomeScreenProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);
  const expandAnim = useState(new Animated.Value(0))[0];

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
    Animated.spring(expandAnim, {
      toValue: isExpanded ? 0 : 1,
      useNativeDriver: false,
    }).start();
  };

  const handleSend = () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText.trim(),
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');

    // TODO: Replace with actual AI response when model is ready
    const mockAiResponse: Message = {
      id: (Date.now() + 1).toString(),
      text: "I'm your AI trainer! Once integrated, I'll provide personalized fitness and nutrition advice based on research.",
      sender: 'ai',
      timestamp: new Date(),
    };

    setTimeout(() => {
      setMessages(prev => [...prev, mockAiResponse]);
    }, 1000);
  };

  const chatHeight = expandAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [80, 400],
  });

  return (
    <View style={styles.container}>
      {/* Main Content */}
      <View style={styles.header}>
        <LogoImage size={120} withText={true} style={styles.logo} />
        <Text style={styles.welcomeText}>Welcome, {userData.name}!</Text>
        <Text style={styles.subtitleText}>Let's crush your goals today!</Text>
      </View>

      <ScrollView>
        <View style={styles.statsContainer}>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>2,400</Text>
            <Text style={styles.statLabel}>Calories</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>180g</Text>
            <Text style={styles.statLabel}>Protein</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>5/6</Text>
            <Text style={styles.statLabel}>Meals</Text>
          </View>
        </View>

        <View style={styles.infoCard}>
          <Text style={styles.cardTitle}>Daily Progress</Text>
          <View style={styles.progressBar}>
            <View style={[styles.progressFill, { width: '75%' }]} />
          </View>
          <Text style={styles.progressText}>75% of daily goals completed</Text>
        </View>

        <View style={styles.infoCard}>
          <Text style={styles.cardTitle}>Today's Tip</Text>
          <Text style={styles.tipText}>
            Remember to stay hydrated! Aim for at least 8 glasses of water today.
          </Text>
        </View>
      </ScrollView>

      {/* Collapsible Chat Interface */}
      <Animated.View style={[styles.collapsibleChat, { height: chatHeight }]}>
        <TouchableOpacity 
          style={styles.chatHeader} 
          onPress={toggleExpand}
          activeOpacity={0.7}
        >
          <Text style={styles.chatTitle}>AI Trainer</Text>
          <Ionicons 
            name={isExpanded ? "chevron-down" : "chevron-up"} 
            size={24} 
            color={COLORS.card} 
          />
        </TouchableOpacity>

        {isExpanded && (
          <>
            <ScrollView style={styles.messagesContainer}>
              {messages.map(message => (
                <View
                  key={message.id}
                  style={[
                    styles.messageContainer,
                    message.sender === 'user' ? styles.userMessage : styles.aiMessage,
                  ]}
                >
                  <Text style={styles.messageText}>{message.text}</Text>
                  <Text style={styles.timestamp}>
                    {message.timestamp.toLocaleTimeString()}
                  </Text>
                </View>
              ))}
            </ScrollView>

            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                value={inputText}
                onChangeText={setInputText}
                placeholder="Ask your AI trainer..."
                placeholderTextColor={COLORS.subtext}
              />
              <TouchableOpacity
                style={[styles.sendButton, !inputText.trim() && styles.sendButtonDisabled]}
                onPress={handleSend}
                disabled={!inputText.trim()}
              >
                <Ionicons name="send" size={20} color={COLORS.card} />
              </TouchableOpacity>
            </View>
          </>
        )}
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  chatContainer: {
    flex: 1,
    padding: 16,
  },
  messageContainer: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 12,
    marginBottom: 8,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: COLORS.primary,
  },
  aiMessage: {
    alignSelf: 'flex-start',
    backgroundColor: COLORS.card,
  },
  messageText: {
    color: COLORS.text,
    fontSize: SIZES.medium,
  },
  timestamp: {
    color: COLORS.subtext,
    fontSize: SIZES.xSmall,
    marginTop: 4,
    alignSelf: 'flex-end',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: COLORS.card,
    ...SHADOWS.medium,
  },
  input: {
    flex: 1,
    backgroundColor: COLORS.background,
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    color: COLORS.text,
    fontSize: SIZES.medium,
  },
  sendButton: {
    backgroundColor: COLORS.primary,
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: COLORS.subtext,
  },
  header: {
    padding: 20,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    ...SHADOWS.medium,
  },
  logo: {
    marginBottom: 10,
  },
  welcomeText: {
    fontSize: SIZES.xLarge,
    fontWeight: 'bold',
    color: COLORS.card,
    marginBottom: 5,
  },
  subtitleText: {
    fontSize: SIZES.medium,
    color: COLORS.card,
    opacity: 0.8,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 15,
    marginTop: 20,
  },
  statCard: {
    backgroundColor: COLORS.card,
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    width: '30%',
    ...SHADOWS.small,
  },
  statValue: {
    fontSize: SIZES.large,
    fontWeight: 'bold',
    color: COLORS.primary,
  },
  statLabel: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
    marginTop: 5,
  },
  infoCard: {
    backgroundColor: COLORS.card,
    margin: 15,
    marginTop: 5,
    borderRadius: 15,
    padding: 20,
    ...SHADOWS.small,
  },
  cardTitle: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 15,
  },
  progressBar: {
    height: 12,
    backgroundColor: COLORS.border,
    borderRadius: 6,
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: COLORS.secondary,
    borderRadius: 6,
  },
  progressText: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
  },
  tipText: {
    fontSize: SIZES.medium,
    color: COLORS.text,
    lineHeight: 22,
  },
  chatHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.card + '40',
  },
  chatTitle: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  messagesContainer: {
    flex: 1,
    padding: 12,
  },
  collapsibleChat: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: COLORS.primary,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    ...SHADOWS.medium,
  },
});

export default HomeScreen; 