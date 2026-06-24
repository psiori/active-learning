import axios from 'axios'
import { useAxios } from '@vueuse/integrations/useAxios'

export const apiClient = axios.create({
  baseURL: '/api/al',
  headers: {
    'Content-Type': 'application/json',
  },
})

function buildErrorMessage(error, fallbackMessage) {
  return error?.response?.data?.detail
    || error?.message
    || fallbackMessage
}

export async function apiRequest(config, fallbackMessage = 'Request failed') {
  const { data, error, execute, response } = useAxios(
    config.url,
    {
      method: config.method ?? 'GET',
      data: config.data,
      params: config.params,
      responseType: config.responseType,
      headers: config.headers,
    },
    apiClient,
    {
      immediate: false,
      resetOnExecute: true,
    },
  )

  await execute()

  if (error.value) {
    throw new Error(buildErrorMessage(error.value, fallbackMessage))
  }

  return {
    data: data.value,
    response: response.value,
  }
}
