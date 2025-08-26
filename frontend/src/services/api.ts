// API 服务层 - 处理与后端的所有通信
import axios, { AxiosResponse } from 'axios';
import type { 
  ApiResponse, 
  DetectionResult, 
  SystemStatus, 
  UploadResponse, 
  HealthCheckResponse 
} from '@types/index';

// 创建 axios 实例
const api = axios.create({
  baseURL: '/api',
  timeout: 60000, // 60秒超时，因为图像处理可能需要较长时间
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证token等
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.config.url} - ${response.status}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    
    // 处理不同类型的错误
    if (error.response?.status === 404) {
      throw new Error('API端点不存在');
    } else if (error.response?.status >= 500) {
      throw new Error('服务器内部错误，请稍后重试');
    } else if (error.response?.status === 413) {
      throw new Error('上传文件过大，请选择较小的图像文件');
    } else if (error.code === 'ECONNABORTED') {
      throw new Error('请求超时，请检查网络连接或稍后重试');
    }
    
    return Promise.reject(error);
  }
);

// 健康检查API
export const healthCheckApi = async (): Promise<HealthCheckResponse> => {
  const response: AxiosResponse<SystemStatus> = await api.get('/health');
  return {
    success: true,
    data: response.data,
    timestamp: new Date().toISOString(),
  };
};

// 简单版本变化检测API
export const uploadAndCompareApi = async (
  image1: File,
  image2: File,
  description?: string
): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('image1', image1);
  formData.append('image2', image2);
  
  if (description) {
    formData.append('description', description);
  }

  const response: AxiosResponse<DetectionResult> = await api.post(
    '/upload-and-compare',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return {
    success: true,
    data: response.data,
    timestamp: new Date().toISOString(),
  };
};

// 高级版本变化检测API
export const uploadAndCompareAdvancedApi = async (
  image1: File,
  image2: File,
  description?: string,
  useAdvanced: boolean = true
): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('image1', image1);
  formData.append('image2', image2);
  formData.append('use_advanced', useAdvanced.toString());
  
  if (description) {
    formData.append('description', description);
  }

  const response: AxiosResponse<DetectionResult> = await api.post(
    '/upload-and-compare-advanced',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return {
    success: true,
    data: response.data,
    timestamp: new Date().toISOString(),
  };
};

// 获取检测结果API
export const getResultApi = async (taskId: string): Promise<ApiResponse<DetectionResult>> => {
  const response: AxiosResponse<DetectionResult> = await api.get(`/result/${taskId}`);
  
  return {
    success: true,
    data: response.data,
    timestamp: new Date().toISOString(),
  };
};

// 图像上传进度回调类型
export interface UploadProgressCallback {
  (progressEvent: { loaded: number; total: number; percentage: number }): void;
}

// 带进度的上传API
export const uploadWithProgressApi = async (
  image1: File,
  image2: File,
  description?: string,
  useAdvanced: boolean = true,
  onProgress?: UploadProgressCallback
): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('image1', image1);
  formData.append('image2', image2);
  formData.append('use_advanced', useAdvanced.toString());
  
  if (description) {
    formData.append('description', description);
  }

  const response: AxiosResponse<DetectionResult> = await api.post(
    useAdvanced ? '/upload-and-compare-advanced' : '/upload-and-compare',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentage = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress({
            loaded: progressEvent.loaded,
            total: progressEvent.total,
            percentage,
          });
        }
      },
    }
  );

  return {
    success: true,
    data: response.data,
    timestamp: new Date().toISOString(),
  };
};

// 文件验证工具函数
export const validateImageFile = (file: File): { valid: boolean; error?: string } => {
  const maxSize = 10 * 1024 * 1024; // 10MB
  const supportedFormats = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/tif'];
  
  if (file.size > maxSize) {
    return {
      valid: false,
      error: `文件大小不能超过 ${maxSize / 1024 / 1024}MB，当前文件大小：${(file.size / 1024 / 1024).toFixed(2)}MB`,
    };
  }
  
  if (!supportedFormats.includes(file.type.toLowerCase())) {
    return {
      valid: false,
      error: `不支持的文件格式：${file.type}，支持的格式：JPG, PNG, TIFF`,
    };
  }
  
  return { valid: true };
};

// 批量文件验证
export const validateImageFiles = (files: File[]): { valid: boolean; errors: string[] } => {
  const errors: string[] = [];
  
  files.forEach((file, index) => {
    const validation = validateImageFile(file);
    if (!validation.valid && validation.error) {
      errors.push(`文件 ${index + 1}: ${validation.error}`);
    }
  });
  
  return {
    valid: errors.length === 0,
    errors,
  };
};

// 获取图像文件信息
export const getImageFileInfo = (file: File): Promise<{
  width: number;
  height: number;
  dataUrl: string;
}> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
        
        resolve({
          width: img.width,
          height: img.height,
          dataUrl,
        });
      } else {
        reject(new Error('无法获取canvas上下文'));
      }
    };
    
    img.onerror = () => {
      reject(new Error('无法加载图像文件'));
    };
    
    img.src = URL.createObjectURL(file);
  });
};

// 压缩图像文件（用于预览）
export const compressImageFile = (
  file: File, 
  maxWidth: number = 1024, 
  maxHeight: number = 1024, 
  quality: number = 0.8
): Promise<{ file: File; dataUrl: string }> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    img.onload = () => {
      // 计算缩放比例
      let { width, height } = img;
      
      if (width > maxWidth || height > maxHeight) {
        const ratio = Math.min(maxWidth / width, maxHeight / height);
        width *= ratio;
        height *= ratio;
      }
      
      canvas.width = width;
      canvas.height = height;
      
      if (ctx) {
        ctx.drawImage(img, 0, 0, width, height);
        
        canvas.toBlob(
          (blob) => {
            if (blob) {
              const compressedFile = new File([blob], file.name, {
                type: 'image/jpeg',
                lastModified: Date.now(),
              });
              
              resolve({
                file: compressedFile,
                dataUrl: canvas.toDataURL('image/jpeg', quality),
              });
            } else {
              reject(new Error('图像压缩失败'));
            }
          },
          'image/jpeg',
          quality
        );
      } else {
        reject(new Error('无法获取canvas上下文'));
      }
    };
    
    img.onerror = () => {
      reject(new Error('无法加载图像文件'));
    };
    
    img.src = URL.createObjectURL(file);
  });
};

// 错误处理工具
export const handleApiError = (error: any): string => {
  if (error.response?.data?.detail) {
    return error.response.data.detail;
  } else if (error.message) {
    return error.message;
  } else {
    return '未知错误，请稍后重试';
  }
};

// 导出所有API函数
export default {
  healthCheck: healthCheckApi,
  uploadAndCompare: uploadAndCompareApi,
  uploadAndCompareAdvanced: uploadAndCompareAdvancedApi,
  getResult: getResultApi,
  uploadWithProgress: uploadWithProgressApi,
  validateImageFile,
  validateImageFiles,
  getImageFileInfo,
  compressImageFile,
  handleApiError,
};