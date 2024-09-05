from __future__ import annotations

## When running alone, you need to add
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import warnings
from typing import Dict

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.pydantic_v1 import Extra, root_validator
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
import requests
from typing import List, Any, Optional
from datetime import datetime
from langchain.prompts import PromptTemplate
from server.agent import model_container
from pydantic import BaseModel, Field

## Use the hefeng Weather API to query the weather
KEY = "ac880e5a877042809ac7ffdd19d95b0d"
# The key looks like this. Here is a sample key. This key cannot be used. You need to register a Hefeng Weather account yourself and then fill in your key here


_PROMPT_TEMPLATE = """
用户会提出一个关于天气的问题，你的目标是拆分出用户问题中的区，市 并按照我提供的工具回答。
例如 用户提出的问题是: 上海浦东未来1小时天气情况？
则 提取的市和区是: 上海 浦东
如果用户提出的问题是: 上海未来1小时天气情况？
则 提取的市和区是: 上海 None
请注意以下内容:
1. 如果你没有找到区的内容,则一定要使用 None 替代，否则程序无法运行
2. 如果用户没有指定市 则直接返回缺少信息

问题: ${{用户的问题}}

你的回答格式应该按照下面的内容，请注意，格式内的```text 等标记都必须输出，这是我用来提取答案的标记。
```text

${{拆分的市和区，中间用空格隔开}}
```
... weathercheck(市 区)...
```output

${{提取后的答案}}
```
答案: ${{答案}}



这是一个例子：
问题: 上海浦东未来1小时天气情况？


```text
上海 浦东
```
...weathercheck(上海 浦东)...

```output
预报时间: 1小时后
具体时间: 今天 18:00
温度: 24°C
天气: 多云
风向: 西南风
风速: 7级
湿度: 88%
降水概率: 16%

Answer: 上海浦东一小时后的天气是多云。

现在，这是我的问题：

问题: {question}
"""
PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)


def get_city_info(location, adm, key):
    base_url = 'https://geoapi.qweather.com/v2/city/lookup?'
    params = {'location': location, 'adm': adm, 'key': key}
    response = requests.get(base_url, params=params)
    data = response.json()
    return data


def format_weather_data(data, place):
    hourly_forecast = data['hourly']
    formatted_data = f"\n This is the weather information for {place} in the next 24 hours: \n"
    for forecast in hourly_forecast:
        # Convert the forecast time to a datetime object
        forecast_time = datetime.strptime(forecast['fxTime'], '%Y-%m-%dT%H:%M%z')
        # Get the time zone of the forecast time
        forecast_tz = forecast_time.tzinfo
        # Get the current time (using the forecast time's time zone)
        now = datetime.now(forecast_tz)
        # Calculate the difference between the forecast date and the current date
        days_diff = (forecast_time.date() - now.date()).days
        if days_diff == 0:
            forecast_date_str = '今天'
        elif days_diff == 1:
            forecast_date_str = '明天'
        elif days_diff == 2:
            forecast_date_str = '后天'
        else:
            forecast_date_str = str(days_diff) + '天后'
        forecast_time_str = forecast_date_str + ' ' + forecast_time.strftime('%H:%M')
        # Calculate the difference between the forecast time and the current time
        time_diff = forecast_time - now
        # Convert the difference to hours
        hours_diff = time_diff.total_seconds() // 3600
        if hours_diff < 1:
            hours_diff_str = '1小时后'
        elif hours_diff >= 24:
            # If more than 24 hours, convert to days
            days_diff = hours_diff // 24
            hours_diff_str = str(int(days_diff)) + '天'
        else:
            hours_diff_str = str(int(hours_diff)) + '小时'
        # Add the difference between the forecast time and the current time to the output
        formatted_data += '预报时间: ' + forecast_time_str + '  距离现在有: ' + hours_diff_str + '\n'
        formatted_data += '温度: ' + forecast['temp'] + '°C\n'
        formatted_data += '天气: ' + forecast['text'] + '\n'
        formatted_data += '风向: ' + forecast['windDir'] + '\n'
        formatted_data += '风速: ' + forecast['windSpeed'] + '级\n'
        formatted_data += '湿度: ' + forecast['humidity'] + '%\n'
        formatted_data += '降水概率: ' + forecast['pop'] + '%\n'
        # formatted_data += '降水量: ' + forecast['precip'] + 'mm\n'
        formatted_data += '\n'
    return formatted_data


def get_weather(key, location_id, place):
    url = "https://devapi.qweather.com/v7/weather/24h?"
    params = {
        'location': location_id,
        'key': key,
    }
    response = requests.get(url, params=params)
    data = response.json()
    return format_weather_data(data, place)


def split_query(query):
    parts = query.split()
    adm = parts[0]
    if len(parts) == 1:
        return adm, adm
    location = parts[1] if parts[1] != 'None' else adm
    return location, adm


def weather(query):
    location, adm = split_query(query)
    key = KEY
    if key == "":
        return "Please fill in the Hefeng Weather API Key in the code first"
    try:
        city_info = get_city_info(location=location, adm=adm, key=key)
        location_id = city_info['location'][0]['id']
        place = adm + "市" + location + "区"

        weather_data = get_weather(key=key, location_id=location_id, place=place)
        return weather_data + "The above is the weather information found, please check\n"
    except KeyError:
        try:
            city_info = get_city_info(location=adm, adm=adm, key=key)
            location_id = city_info['location'][0]['id']
            place = adm + "市"
            weather_data = get_weather(key=key, location_id=location_id, place=place)
            return weather_data + "Important reminder: The city and district information provided by the user does not exist or contains typos, so this information is about the city's weather. Please check\n"
        except KeyError:
            return "The entered area does not exist and no weather forecast can be provided."


class LLMWeatherChain(Chain):
    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    prompt: BasePromptTemplate = PROMPT
    """[Deprecated] Prompt to use to translate to python if necessary."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMWeatherChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _evaluate_expression(self, expression: str) -> str:
        try:
            output = weather(expression)
        except Exception as e:
            output = "The information you entered is incorrect, please try again"
        return output

    def _process_llm_result(
            self, llm_output: str, run_manager: CallbackManagerForChainRun
    ) -> Dict[str, str]:

        run_manager.on_text(llm_output, color="green", verbose=self.verbose)

        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            return {self.output_key: f"The input format is incorrect: {llm_output}, you should enter the combination of (city district)"}
        return {self.output_key: answer}

    async def _aprocess_llm_result(
            self,
            llm_output: str,
            run_manager: AsyncCallbackManagerForChainRun,
    ) -> Dict[str, str]:
        await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)

        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            await run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            await run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {llm_output}")
        return {self.output_key: answer}

    def _call(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs[self.input_key])
        llm_output = self.llm_chain.predict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return self._process_llm_result(llm_output, _run_manager)

    async def _acall(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(inputs[self.input_key])
        llm_output = await self.llm_chain.apredict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return await self._aprocess_llm_result(llm_output, _run_manager)

    @property
    def _chain_type(self) -> str:
        return "llm_weather_chain"

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: BasePromptTemplate = PROMPT,
            **kwargs: Any,
    ) -> LLMWeatherChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)


def weathercheck(query: str):
    model = model_container.MODEL
    llm_weather = LLMWeatherChain.from_llm(model, verbose=True, prompt=PROMPT)
    ans = llm_weather.run(query)
    return ans


class WhetherSchema(BaseModel):
    location: str = Field(description="It should be the name of a region, separated by spaces, for example: Shanghai Pudong. If there is no district information, you can just enter Shanghai")

if __name__ == '__main__':
    result = weathercheck("苏州姑苏区今晚热不热？")
