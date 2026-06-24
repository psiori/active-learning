import axios from 'axios'

const samaClient = axios.create({
  baseURL: '/api/sama',
  headers: { 'Content-Type': 'application/json' },
})

function detail(error, fallback) {
  return error?.response?.data?.detail || error?.message || fallback
}

async function call(config, fallback = 'Request failed') {
  try {
    const { data } = await samaClient.request(config)
    return data
  } catch (error) {
    throw new Error(detail(error, fallback))
  }
}

export async function fetchSamaProjects() {
  return call({ url: '/projects', method: 'GET' }, 'Failed to load Sama projects')
}

export async function fetchSamaBatches({ origins, days = 90, enrich = true } = {}) {
  return call(
    {
      url: '/batches',
      method: 'GET',
      params: {
        origins: origins && origins.length ? origins.join(',') : undefined,
        days,
        enrich,
      },
    },
    'Failed to load Sama batches',
  )
}

export async function startImportJob(payload) {
  return call({ url: '/jobs/import', method: 'POST', data: payload }, 'Import failed to start')
}

export async function startMaskJob(payload) {
  return call({ url: '/jobs/masks', method: 'POST', data: payload }, 'Mask job failed to start')
}

export async function startPipelineJob(payload) {
  return call(
    { url: '/jobs/pipeline', method: 'POST', data: payload },
    'Pipeline failed to start',
  )
}

export async function approveTopic(dataset_topic, approved) {
  return call(
    { url: '/approve', method: 'POST', data: { dataset_topic, approved } },
    'Approval failed',
  )
}

export async function fetchTopicState(dataset_topic) {
  return call(
    { url: `/topic/${encodeURIComponent(dataset_topic)}`, method: 'GET' },
    'Failed to load topic state',
  )
}

export function mosaicUrl(dataset_topic) {
  return `/api/sama/mosaic/${encodeURIComponent(dataset_topic)}`
}

export function openJobEventStream(jobId, { onEvent, onEnd, onError } = {}) {
  const source = new EventSource(`/api/al/jobs/${encodeURIComponent(jobId)}/events`)
  const eventNames = ['snapshot', 'status', 'progress', 'result', 'error', 'end']
  for (const name of eventNames) {
    source.addEventListener(name, (event) => {
      try {
        const data = JSON.parse(event.data)
        onEvent?.(name, data)
        if (name === 'end') {
          source.close()
          onEnd?.(data)
        }
      } catch (parseError) {
        onError?.(parseError)
      }
    })
  }
  source.onerror = (err) => {
    onError?.(err)
  }
  return source
}
