import tensorflow as tf

import numpy as np
import os
import time

import telebot
from telebot import types
import random
import datetime
import sqlite3
# import all the modules




text = open('tutby_titles.txt', 'rb').read().decode(encoding='utf-8')
# writing all the titles from file to variable 'text'
# doing it line by line cause my server wasn't powerful enough to do it in one time

vocab = sorted(set(text))
# creating vocabulary
print(f'unique characters {len(vocab)}')

char2idx = {u:i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)
# these two are converting characters to numbers and vice versa
# cause neural networks work only with numbers


text_as_int = np.array([char2idx[c] for c in text])
# making np.array out of all the text as integers
# doing it in 150 times cause, again, my server wasn't powerful enough


# The maximum length sentence you want for a single input in characters
seq_length = 265



# creating dataset from array with all letters as integers
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
# It was done for neural network to understand it

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text





# Batch size
BATCH_SIZE = 72

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000




# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# function for builing model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
    batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
    return_sequences=True,
    stateful=True,
    recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
    ])
    return model



# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'



#building model
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)


#loading weights of latest checkpoint
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

#building model
model.build(tf.TensorShape([1, None]))

print(model.summary())


# main func to generate text
def generate_text(model, start_string, temper = 0.45):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = temper

    # Here batch size == 1
    model.reset_states()
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    while idx2char[predicted_id] != '₪':
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

#global vatiables for telegram bot
#cause telebot (module for telegam bots) works via decorators and functions under them
#In python to transfer values between functions with variables we need to make them global
#Otherwise, values will remain local to function
global new_title
new_title = ''

global last_temp
last_temp = 0.53

global last_key
last_key = ''

global last_w_r
last_w_r = []

global want_to_send_report
want_to_send_report = False

global want_to_rply
want_to_rply = False

#connecting with bot
bot = telebot.TeleBot('token')

#keyboard for administrator
admkb = telebot.types.ReplyKeyboardMarkup(True)
admkb.row('/about', 'новые сообщения', 'пользователи')

# connecting with user reports database
conn = sqlite3.connect('user_reports.db')
# cursor to do something with database
cursor = conn.cursor()
# if there is no table in db, create it
cursor.execute('CREATE TABLE IF NOT EXISTS reports (chat TEXT, date TEXT, message_id TEXT, wtchd TEXT)')

#adding to admin list all users that have to check users info
admins_list = [504898099]

#on ready send message to admins that everything is good and bot is running
for admin in admins_list:
    bot.send_message(admin, 'Все готово', reply_markup = admkb)


#func to make inline keyboard for telegram bot
#its in func cause it will be called every new messoge so text can be different
def init_keyb():
  keyboard = types.InlineKeyboardMarkup()# creating keyboard
  key_like = types.InlineKeyboardButton(text=random.choice(['Класс', 'Отлично', 'Вот это хорошее', 'да', 'запомни это', 'ок', 'норм']), callback_data='like') # creating <<like>> button
  key_dis = types.InlineKeyboardButton(text=random.choice(['фе', 'фу', 'ненадо нам больше такого', 'нет', 'бред это все', 'боже что удоли это']), callback_data='bad') # and dislike button
#                                                           /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\
#                                                                  Words that appear in button. Random one
  keyboard.add(key_like, key_dis) # adding these buttons to be in one row
  key_more = types.InlineKeyboardButton(text='ещё', callback_data='escho') # adding button for creating new title
  keyboard.add(key_more) # adding this button
  return keyboard #

keyboard = init_keyb() # initializing keyboard for the first time


adm_rep_kb = types.InlineKeyboardMarkup()
key_read = types.InlineKeyboardButton(text = 'прочитать след.', callback_data = 'read_rep')
key_repl = types.InlineKeyboardButton(text = 'Ответить', callback_data = 'reply_to_rep')
adm_rep_kb.add(key_read, key_repl)

kbkb = telebot.types.ReplyKeyboardMarkup(True)
kbkb.row('/about', 'еще', '/report')

is_closed = False

if is_closed:
    @bot.message_handler(content_types=['text'])
    def busy_with_work(message):
        if message.text.lower() in ['бастуете', 'бастуете?', 'забастовка?', 'забастовка', 'бастуешь?']:
            bot.send_message(message.chat.id, 'Ну да, бастуем')
        else:
            bot.send_message(message.chat.id, 'Ведутся технические работы')



@bot.message_handler(commands = ['start', 'about', 'info', 'report'])
def send_txt_file(message):
    global new_title
    global want_to_send_report
    global last_w_r
    if want_to_send_report:
        return
    if message.text.lower().startswith('/start'):
        bot.send_message(message.chat.id, 'Чтобы начать напиши мне любой текст или нажми кнопку <еще> под постом\n\nЕсли тебе понравится нейроновсть, то нажми левую кнопку(где будет что-то типа класс, гуд, оу и т.д.), А если нет - то на правую(где будет фу, бред и т. д.)\n\nЧтобы получить ещё одну новсть нажми на <ещё>', reply_markup = keyboard)
        with open('user_list.txt', 'r') as r_log_file:
            if message.from_user not in r_log_file:
                open('user_list.txt', 'a').write(f'{message.from_user}\n\n')
    elif message.text.lower().startswith('/report'):
        if len(message.text.split()) > 1:
            conn = sqlite3.connect('user_reports.db')
            cursor = conn.cursor()
            report_data = [str(message.chat), str(datetime.datetime.now()), str(message.chat.id) + ';' + str(message.message_id), False]
            cursor.executemany('INSERT INTO reports VALUES (?, ?, ?, ?)', (report_data, ))
            bot.send_message(message.chat.id, 'Ваше заявление отправлено', reply_markup = kbkb)
            conn.commit()
        else:
            bot.send_message(message.chat.id, 'Пожалуйста отправьте сообщение, которое вы хотите передать админисратору. Я вас слушаю')
            want_to_send_report = True
    else:
        bot.send_message(message.chat.id, 'Этот бот генерирует заголовки новостей\n\nс переменным успехом, но все же\n\nЧтобы сгенерировать нейроновость воспользуйтесь кнопками под сообщением\n\nДля генерации по первым словам ставте «!» перед сообщением\n\nЕсли вы отправите десятичное число(например 0.54 или 3.72), то это изменит генерацию сообщений. Чем больше число, тем более абсурдный и неправдоподобный результат, чем меньше число - наоборот, новости более топорные\n\nДля связи с администрацией используйте /report и после пишете сообщение.\nЛибо /report <Сообщение>', reply_markup = keyboard)
    new_title = ''


@bot.message_handler(commands = ['file'])
def send_txt_file(message):
    global new_title
    global want_to_send_report
    if want_to_send_report == False:
        if message.chat.id == 504898099:
            if message.text.split()[1:] in os.listdir():
                bot.send_document(message.chat.id, open(message.text.split()[1:], 'r'))
            else:
                bot.send_document(message.chat.id, open('phrases_logs.txt', 'r'))
    new_title = ''


@bot.message_handler(content_types=['text'])
def send_neuro_new(message):
  keyboard = init_keyb()
  global new_title
  global last_temp
  global last_key
  global last_w_r
  global want_to_send_report
  global want_to_rply
  conn = sqlite3.connect('user_reports.db')
  cursor = conn.cursor()
  print(message.text)
  if want_to_send_report:
      report_data = [str(message.chat), str(datetime.datetime.now()), str(message.chat.id) + ';' + str(message.message_id), False]
      cursor.executemany('INSERT INTO reports VALUES (?, ?, ?, ?)', (report_data, ))
      bot.send_message(message.chat.id, 'Ваше заявление отправлено', reply_markup = kbkb)
      conn.commit()
      want_to_send_report = False
      return
  elif want_to_rply:
      bot.send_message(last_w_r[0], f'Вам пришел ответ от адмнистрации на сообщение от {last_w_r[1][:18]}:\n' + message.text)
      bot.send_message(message.chat.id, 'Сообщение отправлено')
      want_to_rply = False
      return
  else:
      pass
  if message.text.lower() == 'новые сообщения' and message.chat.id == 504898099:
      cursor.execute('SELECT * FROM reports WHERE wtchd = FALSE')
      num_of_reps = 0
      new_reps = cursor.fetchall()
      for reprsrs in new_reps:
          num_of_reps += 1
      bot.send_message(message.chat.id, f'У вас есть [{num_of_reps}] нов. сообщений', reply_markup = adm_rep_kb)
      return
  elif message.text.lower() == 'пользователи' and message.chat.id == 504898099:
      bot.send_document(message.chat.id, open('user_list.txt', 'r'), caption = 'Файл, со всеми пользователями, нажавшими старт')
      return
  try:
      last_temp = float(message.text)
      new_title = generate_text(model, start_string=f"♣{last_key}", temper = float(message.text))
  except ValueError:
      if message.text.startswith('!'):
          last_key = message.text[1:]
          if 'Й' in message.text:
              message.text.replace('Й', 'й')
          if message.text.startswith('!Й'):
              new_title = generate_text(model, start_string=f"♣{message.text.replace('!', '♣').replace('Й', 'й')}")
              new_title = '♣' + 'Й' + new_title.replace('♣', '')[1:]
          else:
              new_title = generate_text(model, start_string=f"♣{last_key}")
      else:
          new_title = generate_text(model, start_string="♣")
          last_key = ''
  bot.send_message(message.chat.id, new_title.replace('♣', '').replace('₪', ''), reply_markup=keyboard)
  with open('phrases_logs.txt', 'a') as log_file:
      log_file.write(f'{new_title} [{last_temp}]\n\n')

@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    keyboard = init_keyb()
    global new_title
    global last_temp
    global want_to_rply
    global last_w_r
    if call.data == "escho":
        new_title = generate_text(model, start_string=u"♣", temper = last_temp)
        bot.send_message(call.message.chat.id, new_title.replace('♣', '').replace('₪', ''), reply_markup=keyboard)
        print(call.message.text)
        print(last_temp)
        with open('phrases_logs.txt', 'a') as log_file:
            log_file.write(f'{new_title} [{last_temp}]\n\n')
    elif call.data == "like":
        if new_title:
            with open('new_tutby_titles.txt', 'a') as sess_file:
                sess_file.write(f'{new_title}\n\n')
                bot.send_message(call.message.chat.id, 'Ок', reply_markup=keyboard)
                new_title = ''
        else:
            bot.send_message(call.message.chat.id, 'Нечего учитывать', reply_markup = keyboard)
    elif call.data == "bad":
        if new_title:
            with open('bad_tutby_titles.txt', 'a') as sess_file:
                sess_file.write(f'{new_title}\n\n')
                bot.send_message(call.message.chat.id, 'Такое больше не повторится(надеюсь)', reply_markup=keyboard)
                new_title = ''
        else:
            bot.send_message(call.message.chat.id, 'Нечего учитывать', reply_markup = keyboard)
    elif call.data == 'read_rep':
        conn = sqlite3.connect('user_reports.db')
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM reports WHERE wtchd = FALSE')
        new_reps = cursor.fetchone()
        if not new_reps:
            bot.send_message(call.message.chat.id, 'Сообщения закончились')
        else:
            reporrt = new_reps
            report_fn = reporrt[0].split(',')[4].replace("'", "").replace('first_name: ', '')
            report_un = reporrt[0].split(',')[3].replace("'", "").replace('username: ', '')
            bot.forward_message(call.message.chat.id, reporrt[2].split(';')[0], reporrt[2].split(';')[1])
            bot.send_message(call.message.chat.id, f"""Получено от {report_fn} {report_un}\nв {reporrt[1][:18]}""", reply_markup = adm_rep_kb)
            last_w_r = reporrt[2].split(';')
            cursor.execute(f'UPDATE reports SET wtchd = TRUE WHERE message_id = ?', (reporrt[2], ))
            conn.commit()
    elif call.data == 'reply_to_rep':
        bot.send_message(call.message.chat.id, 'Я тея слушаю')
        want_to_rply = True

bot.polling()
