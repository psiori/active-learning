import { apiRequest } from './axios'

export async function fetchConfig() {
  const { data } = await apiRequest({ url: '/config' }, 'Failed to load project config')
  return data
}

export async function fetchProject(projectName) {
  const { data } = await apiRequest(
    { url: `/projects/${encodeURIComponent(projectName)}` },
    'Failed to load project details',
  )
  return data
}

export async function createJob(kind, payload) {
  const { data } = await apiRequest(
    {
      url: '/jobs',
      method: 'POST',
      data: { ...payload, kind },
    },
    `${kind} job failed to start`,
  )
  return data
}

export async function fetchJob(jobId) {
  const { data } = await apiRequest(
    { url: `/jobs/${encodeURIComponent(jobId)}` },
    'Failed to load job',
  )
  return data
}

export async function downloadOverlayMosaic(payload) {
  const { response } = await apiRequest(
    {
      url: '/export/overlay-mosaic',
      method: 'POST',
      data: payload,
      responseType: 'blob',
    },
    'Overlay mosaic download failed',
  )

  const blob = response.data
  const url = window.URL.createObjectURL(blob)
  const disposition = response.headers['content-disposition'] || ''
  const filenameMatch = disposition.match(/filename="?([^"]+)"?/)
  const filename = filenameMatch?.[1] || 'overlay-mosaic.jpg'

  return { url, filename }
}

export async function seedStrategySelection(payload) {
  const { data } = await apiRequest(
    {
      url: '/export/seed',
      method: 'POST',
      data: payload,
    },
    'Seed to Sama failed',
  )
  return data
}

export async function cancelJob(jobId) {
  await apiRequest(
    { url: `/jobs/${encodeURIComponent(jobId)}`, method: 'DELETE' },
    'Cancel failed',
  )
}

export async function patchProjectConfig(projectName, payload) {
  const { data } = await apiRequest(
    {
      url: `/projects/${encodeURIComponent(projectName)}`,
      method: 'PATCH',
      data: payload,
    },
    'Failed to save project config',
  )
  return data
}

export async function exportExclusionTags(payload) {
  const { data } = await apiRequest(
    {
      url: '/export/exclusion-tags',
      method: 'POST',
      data: payload,
    },
    'Export exclusion tags failed',
  )
  return data
}
