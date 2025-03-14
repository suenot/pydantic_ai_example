import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Загрузка переменных окружения из файла .env
load_dotenv()

# Получение API ключа из переменных окружения
api_key = os.getenv("OPENROUTER_API_KEY")

# Проверка наличия API ключа
if not api_key:
    raise ValueError("API ключ OpenRouter не найден. Проверьте файл .env")

print(f"API ключ загружен: {api_key[:10]}...")

# Инициализация клиента OpenAI с базовым URL OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    # Дополнительные заголовки для OpenRouter
    default_headers={
        "HTTP-Referer": "http://localhost:8000",  # Необходимо для OpenRouter
        "X-Title": "Pydantic AI Demo",  # Название вашего приложения
    }
)

# Определение Pydantic моделей для структурированных данных
class Person(BaseModel):
    name: str = Field(..., description="Имя человека")
    age: int = Field(..., description="Возраст человека")
    occupation: str = Field(..., description="Профессия человека")
    skills: List[str] = Field(default_factory=list, description="Список навыков")
    appearance: str = Field(..., description="Внешность человека")
    personality: str = Field(..., description="Характер человека")
    history: str = Field(..., description="История человека")
    bio: Optional[str] = Field(None, description="Краткая биография")

# Функция для получения структурированных данных от модели
def get_structured_data(prompt, model_name="google/gemma-3-4b-it:free"):
    system_message = """
    Ты - помощник, который всегда отвечает в формате JSON.
    Твой ответ должен быть валидным JSON объектом, соответствующим следующей схеме:
    {
        "name": "string",
        "age": number,
        "occupation": "string",
        "skills": ["string", "string", ...],
        "appearance": "string",
        "personality": "string",
        "history": "string",
        "bio": "string" (optional)
    }
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        
        # Получаем текст ответа
        response_text = response.choices[0].message.content
        
        # Пытаемся извлечь JSON из ответа
        try:
            # Ищем начало и конец JSON в ответе
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                # Валидация данных с помощью Pydantic
                validated_data = Person(**data)
                return validated_data
            else:
                print("Не удалось найти JSON в ответе модели")
                print("Полученный ответ:", response_text)
                return None
                
        except json.JSONDecodeError as e:
            print(f"Ошибка декодирования JSON: {e}")
            print("Полученный ответ:", response_text)
            return None
        except ValidationError as e:
            print(f"Ошибка валидации данных: {e}")
            return None
            
    except Exception as e:
        print(f"Ошибка при запросе к модели: {e}")
        return None

# Пример использования
if __name__ == "__main__":
    prompt = "Создай профиль вымышленного персонажа девшуки из игр Hoyoverse (Genshin Impact, Honkai Impact 3rd, etc.), которая специализируется на Python и машинном обучении."
    
    person = get_structured_data(prompt)
    
    if person:
        print("\nПолученные данные (валидированы с Pydantic):")
        print(f"Имя: {person.name}")
        print(f"Возраст: {person.age}")
        print(f"Профессия: {person.occupation}")
        print(f"Навыки: {', '.join(person.skills)}")
        print(f"Внешность: {person.appearance}")
        print(f"Характер: {person.personality}")
        print(f"История: {person.history}")
        if person.bio:
            print(f"Биография: {person.bio}")
        
        # Можно также получить данные в формате JSON
        print("\nДанные в формате JSON:")
        print(person.model_dump_json(indent=2)) 
