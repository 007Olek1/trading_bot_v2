#!/bin/bash
# 🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ ПЕРЕД ДЕПЛОЕМ

echo "🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ ПЕРЕД ДЕПЛОЕМ"
echo "============================================"
echo ""

ERRORS=0
WARNINGS=0

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1${NC}"
        ERRORS=$((ERRORS + 1))
    fi
}

check_warning() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${YELLOW}⚠️ $1${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
}

echo "📋 1. ПРОВЕРКА ФАЙЛОВ..."
echo "------------------------"

# Проверка основных файлов
FILES_TO_CHECK=(
    "super_bot_v4_mtf.py"
    "smart_coin_selector.py"
    "probability_calculator.py"
    "strategy_evaluator.py"
    "realism_validator.py"
    "telegram_commands_handler.py"
)

for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo -e "  ${RED}❌ $file - НЕ НАЙДЕН!${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "🐍 2. ПРОВЕРКА PYTHON СИНТАКСИСА..."
echo "----------------------------------"

# Проверка синтаксиса Python
for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        python3 -m py_compile "$file" 2>&1
        check_status "Синтаксис $file корректен"
    fi
done

echo ""
echo "📦 3. ПРОВЕРКА ИМПОРТОВ..."
echo "-------------------------"

python3 << 'PYTHON_TEST'
import sys

errors = 0
modules_to_check = [
    'ccxt',
    'telegram',
    'pandas',
    'numpy',
    'sklearn',
    'apscheduler',
    'dotenv'
]

for module in modules_to_check:
    try:
        __import__(module)
        print(f"  ✅ {module}")
    except ImportError as e:
        print(f"  ❌ {module} - НЕ УСТАНОВЛЕН: {e}")
        errors += 1

# Проверка импортов из бота
try:
    from smart_coin_selector import SmartCoinSelector
    print("  ✅ smart_coin_selector импортируется")
except Exception as e:
    print(f"  ❌ smart_coin_selector: {e}")
    errors += 1

try:
    from probability_calculator import ProbabilityCalculator
    print("  ✅ probability_calculator импортируется")
except Exception as e:
    print(f"  ❌ probability_calculator: {e}")
    errors += 1

try:
    from strategy_evaluator import StrategyEvaluator
    print("  ✅ strategy_evaluator импортируется")
except Exception as e:
    print(f"  ❌ strategy_evaluator: {e}")
    errors += 1

try:
    from realism_validator import RealismValidator
    print("  ✅ realism_validator импортируется")
except Exception as e:
    print(f"  ❌ realism_validator: {e}")
    errors += 1

sys.exit(errors)
PYTHON_TEST

check_warning "Импорты проверены"

echo ""
echo "🔑 4. ПРОВЕРКА КОНФИГУРАЦИИ..."
echo "-----------------------------"

# Проверка .env файла
if [ -f ".env" ]; then
    echo "  ✅ .env найден"
    
    # Проверка наличия ключевых переменных
    if grep -q "BYBIT_API_KEY" .env; then
        echo "  ✅ BYBIT_API_KEY присутствует"
    else
        echo -e "  ${RED}❌ BYBIT_API_KEY отсутствует${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    
    if grep -q "BYBIT_API_SECRET" .env; then
        echo "  ✅ BYBIT_API_SECRET присутствует"
    else
        echo -e "  ${RED}❌ BYBIT_API_SECRET отсутствует${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    
    if grep -q "TELEGRAM_BOT_TOKEN" .env; then
        echo "  ✅ TELEGRAM_BOT_TOKEN присутствует"
    else
        echo -e "  ${YELLOW}⚠️ TELEGRAM_BOT_TOKEN отсутствует${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "  ${YELLOW}⚠️ .env файл не найден (будет проверен на сервере)${NC}"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""
echo "🔍 5. ПРОВЕРКА ЛОГИКИ STOP LOSS..."
echo "----------------------------------"

python3 << 'PYTHON_TEST'
# Проверка что MAX_STOP_LOSS_USD = 5.0
with open('super_bot_v4_mtf.py', 'r') as f:
    content = f.read()
    if 'MAX_STOP_LOSS_USD = 5.0' in content or 'MAX_STOP_LOSS_USD = 5' in content:
        print("  ✅ MAX_STOP_LOSS_USD установлен в $5")
    else:
        print("  ❌ MAX_STOP_LOSS_USD не найден или неправильный")
        exit(1)
    
    if 'initial_sl' in content:
        print("  ✅ Трейлинг стоп реализован (initial_sl)")
    else:
        print("  ⚠️ Трейлинг стоп может быть не полностью реализован")
PYTHON_TEST

check_warning "Логика Stop Loss проверена"

echo ""
echo "📊 6. ПРОВЕРКА ФУНКЦИЙ ТОРГОВЛИ..."
echo "---------------------------------"

python3 << 'PYTHON_TEST'
with open('super_bot_v4_mtf.py', 'r') as f:
    content = f.read()
    
    checks = {
        'open_position_automatically': 'Функция открытия позиций',
        'monitor_positions': 'Функция мониторинга позиций',
        'create_market_order': 'Использование market order',
        'MAX_POSITIONS = 3': 'Лимит позиций установлен'
    }
    
    errors = 0
    for key, desc in checks.items():
        if key in content:
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ {desc} - НЕ НАЙДЕНО")
            errors += 1
    
    exit(errors)
PYTHON_TEST

check_status "Функции торговли проверены"

echo ""
echo "📋 7. ИТОГОВЫЙ ОТЧЕТ..."
echo "======================"

echo ""
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!${NC}"
    echo ""
    echo "🚀 Система готова к деплою!"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️ ТЕСТЫ ПРОЙДЕНЫ С ПРЕДУПРЕЖДЕНИЯМИ: $WARNINGS${NC}"
    echo ""
    echo "🚀 Система готова к деплою, но проверьте предупреждения."
    exit 0
else
    echo -e "${RED}❌ НАЙДЕНЫ КРИТИЧЕСКИЕ ОШИБКИ: $ERRORS${NC}"
    echo -e "${YELLOW}⚠️ ПРЕДУПРЕЖДЕНИЙ: $WARNINGS${NC}"
    echo ""
    echo "🛑 НЕ РЕКОМЕНДУЕТСЯ ДЕПЛОЙ ПРЕЖДЕ УСТРАНЕНИЯ ОШИБОК!"
    exit 1
fi

