-- 🗄️ ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ TRADING BOT V2.0

-- Создание базы данных
CREATE DATABASE trading_bot;

-- Создание пользователя
CREATE USER trading_bot WITH PASSWORD 'trading_bot_password';

-- Права доступа
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_bot;

-- Подключаемся к базе
\c trading_bot

-- Права на схему public
GRANT ALL ON SCHEMA public TO trading_bot;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_bot;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_bot;

-- Включаем расширения
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

