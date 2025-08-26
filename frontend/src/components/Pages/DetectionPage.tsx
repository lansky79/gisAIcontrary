import React, { useState, useCallback } from 'react';
import { 
  Layout, 
  Card, 
  Upload, 
  Button, 
  Progress, 
  message, 
  Space, 
  Row, 
  Col,
  Input,
  Switch,
  Divider,
  Tag,
  Spin,
  Image
} from 'antd';
import { 
  InboxOutlined, 
  CloudUploadOutlined, 
  DeleteOutlined,
  EyeOutlined,
  SettingOutlined
} from '@ant-design/icons';
import type { UploadProps, UploadFile } from 'antd';
import MapComponent from '@components/Map/MapComponent';
import { useAppStore } from '@store/useAppStore';
import { 
  uploadWithProgressApi, 
  validateImageFiles, 
  getImageFileInfo,
  handleApiError 
} from '@services/api';
import type { DetectionResult, ChangeRegion } from '@types/index';

const { Content } = Layout;
const { TextArea } = Input;
const { Dragger } = Upload;

interface UploadedImage {
  file: File;
  preview: string;
  info: {
    width: number;
    height: number;
    size: string;
  };
}

const DetectionPage: React.FC = () => {
  const { currentTask, setCurrentTask, analysisConfig } = useAppStore();
  
  // 上传状态
  const [uploadedImages, setUploadedImages] = useState<{
    image1: UploadedImage | null;
    image2: UploadedImage | null;
  }>({ image1: null, image2: null });
  
  // 处理状态
  const [processing, setProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [description, setDescription] = useState('');
  const [useAdvanced, setUseAdvanced] = useState(true);
  
  // 结果状态
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<ChangeRegion | null>(null);

  // 处理文件上传
  const handleFileUpload = useCallback(async (file: File, imageKey: 'image1' | 'image2') => {
    try {
      // 验证文件
      const validation = validateImageFiles([file]);
      if (!validation.valid) {
        message.error(validation.errors[0]);
        return false;
      }

      // 获取图像信息和预览
      const imageInfo = await getImageFileInfo(file);
      
      const uploadedImage: UploadedImage = {
        file,
        preview: imageInfo.dataUrl,
        info: {
          width: imageInfo.width,
          height: imageInfo.height,
          size: `${(file.size / 1024 / 1024).toFixed(2)}MB`,
        }
      };

      setUploadedImages(prev => ({
        ...prev,
        [imageKey]: uploadedImage,
      }));

      message.success(`${imageKey === 'image1' ? '历史图像' : '新图像'}上传成功`);
      return false; // 阻止自动上传
    } catch (error) {
      message.error('图像处理失败: ' + handleApiError(error));
      return false;
    }
  }, []);

  // 移除已上传的图像
  const handleRemoveImage = (imageKey: 'image1' | 'image2') => {
    setUploadedImages(prev => ({
      ...prev,
      [imageKey]: null,
    }));
  };

  // 执行变化检测
  const handleStartDetection = async () => {
    if (!uploadedImages.image1 || !uploadedImages.image2) {
      message.error('请先上传两张图像');
      return;
    }

    setProcessing(true);
    setUploadProgress(0);
    setDetectionResult(null);

    try {
      const response = await uploadWithProgressApi(
        uploadedImages.image1.file,
        uploadedImages.image2.file,
        description || undefined,
        useAdvanced,
        (progress) => {
          setUploadProgress(progress.percentage);
        }
      );

      if (response.success && response.data) {
        setDetectionResult(response.data);
        message.success('变化检测完成！');
        
        // 如果有GPS信息，自动调整地图视图
        if (response.data.gpsInfo.image1 || response.data.gpsInfo.image2) {
          const gps = response.data.gpsInfo.image1 || response.data.gpsInfo.image2;
          if (gps) {
            // 这里可以更新地图中心点
            console.log('GPS信息:', gps);
          }
        }
      } else {
        throw new Error('检测失败');
      }
    } catch (error) {
      message.error('变化检测失败: ' + handleApiError(error));
    } finally {
      setProcessing(false);
      setUploadProgress(0);
    }
  };

  // 清空所有内容
  const handleClear = () => {
    setUploadedImages({ image1: null, image2: null });
    setDetectionResult(null);
    setSelectedRegion(null);
    setDescription('');
    setUploadProgress(0);
  };

  // 处理区域选择
  const handleRegionSelect = (region: ChangeRegion) => {
    setSelectedRegion(region);
    message.info(`选中变化区域: ${region.changeType} (置信度: ${(region.confidence * 100).toFixed(1)}%)`);
  };

  // 上传组件属性
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: false,
    accept: 'image/*',
    showUploadList: false,
    beforeUpload: () => false, // 阻止自动上传
  };

  return (
    <Layout style={{ height: '100%', background: '#f5f5f5' }}>
      <Content style={{ padding: 16, overflow: 'auto' }}>
        <Row gutter={16} style={{ height: '100%' }}>
          {/* 左侧控制面板 */}
          <Col span={8}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              {/* 图像上传区域 */}
              <Card title="图像上传" size="small">
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  {/* 历史图像上传 */}
                  <div>
                    <div style={{ marginBottom: 8, fontWeight: 500 }}>历史基准图像</div>
                    {uploadedImages.image1 ? (
                      <Card 
                        size="small" 
                        bodyStyle={{ padding: 8 }}
                        actions={[
                          <EyeOutlined key="preview" onClick={() => {
                            Image.preview({
                              src: uploadedImages.image1!.preview,
                            });
                          }} />,
                          <DeleteOutlined key="delete" onClick={() => handleRemoveImage('image1')} />
                        ]}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <img 
                            src={uploadedImages.image1.preview} 
                            alt="历史图像" 
                            style={{ width: 60, height: 60, objectFit: 'cover', borderRadius: 4 }}
                          />
                          <div style={{ flex: 1, fontSize: 12 }}>
                            <div>{uploadedImages.image1.file.name}</div>
                            <div style={{ color: '#666' }}>
                              {uploadedImages.image1.info.width} × {uploadedImages.image1.info.height}
                            </div>
                            <div style={{ color: '#666' }}>{uploadedImages.image1.info.size}</div>
                          </div>
                        </div>
                      </Card>
                    ) : (
                      <Dragger 
                        {...uploadProps}
                        onChange={(info) => {
                          if (info.file.originFileObj) {
                            handleFileUpload(info.file.originFileObj, 'image1');
                          }
                        }}
                        style={{ padding: '20px 10px' }}
                      >
                        <p className="ant-upload-drag-icon">
                          <InboxOutlined style={{ fontSize: 24 }} />
                        </p>
                        <p className="ant-upload-text">点击或拖拽上传历史图像</p>
                        <p className="ant-upload-hint">支持 JPG, PNG, TIFF 格式</p>
                      </Dragger>
                    )}
                  </div>

                  {/* 新图像上传 */}
                  <div>
                    <div style={{ marginBottom: 8, fontWeight: 500 }}>新拍摄图像</div>
                    {uploadedImages.image2 ? (
                      <Card 
                        size="small" 
                        bodyStyle={{ padding: 8 }}
                        actions={[
                          <EyeOutlined key="preview" onClick={() => {
                            Image.preview({
                              src: uploadedImages.image2!.preview,
                            });
                          }} />,
                          <DeleteOutlined key="delete" onClick={() => handleRemoveImage('image2')} />
                        ]}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <img 
                            src={uploadedImages.image2.preview} 
                            alt="新图像" 
                            style={{ width: 60, height: 60, objectFit: 'cover', borderRadius: 4 }}
                          />
                          <div style={{ flex: 1, fontSize: 12 }}>
                            <div>{uploadedImages.image2.file.name}</div>
                            <div style={{ color: '#666' }}>
                              {uploadedImages.image2.info.width} × {uploadedImages.image2.info.height}
                            </div>
                            <div style={{ color: '#666' }}>{uploadedImages.image2.info.size}</div>
                          </div>
                        </div>
                      </Card>
                    ) : (
                      <Dragger 
                        {...uploadProps}
                        onChange={(info) => {
                          if (info.file.originFileObj) {
                            handleFileUpload(info.file.originFileObj, 'image2');
                          }
                        }}
                        style={{ padding: '20px 10px' }}
                      >
                        <p className="ant-upload-drag-icon">
                          <InboxOutlined style={{ fontSize: 24 }} />
                        </p>
                        <p className="ant-upload-text">点击或拖拽上传新图像</p>
                        <p className="ant-upload-hint">支持 JPG, PNG, TIFF 格式</p>
                      </Dragger>
                    )}
                  </div>
                </Space>
              </Card>

              {/* 检测配置 */}
              <Card title="检测配置" size="small">
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  <div>
                    <div style={{ marginBottom: 8, fontWeight: 500 }}>检测描述（可选）</div>
                    <TextArea
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="描述需要检测的变化类型，如：新建筑物、道路变化、植被变化等..."
                      rows={3}
                    />
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 500 }}>使用高级算法</span>
                    <Switch
                      checked={useAdvanced}
                      onChange={setUseAdvanced}
                      checkedChildren="高级"
                      unCheckedChildren="简单"
                    />
                  </div>

                  {useAdvanced && (
                    <div style={{ fontSize: 12, color: '#666', background: '#f8f9fa', padding: 8, borderRadius: 4 }}>
                      高级算法提供更精确的图像配准和智能变化检测，但处理时间稍长
                    </div>
                  )}
                </Space>
              </Card>

              {/* 操作按钮 */}
              <Card size="small">
                <Space style={{ width: '100%' }} direction="vertical">
                  {processing && (
                    <div>
                      <div style={{ marginBottom: 8 }}>
                        <Spin size="small" /> 正在处理中...
                      </div>
                      <Progress percent={uploadProgress} size="small" />
                    </div>
                  )}
                  
                  <Space style={{ width: '100%' }}>
                    <Button
                      type="primary"
                      icon={<CloudUploadOutlined />}
                      onClick={handleStartDetection}
                      disabled={!uploadedImages.image1 || !uploadedImages.image2 || processing}
                      loading={processing}
                      style={{ flex: 1 }}
                    >
                      开始检测
                    </Button>
                    
                    <Button
                      icon={<DeleteOutlined />}
                      onClick={handleClear}
                      disabled={processing}
                    >
                      清空
                    </Button>
                  </Space>
                </Space>
              </Card>

              {/* 检测结果摘要 */}
              {detectionResult && (
                <Card title="检测结果" size="small">
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>变化区域数量:</span>
                      <Tag color="blue">{detectionResult.detectionResults.changeRegionsCount}</Tag>
                    </div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>变化面积比例:</span>
                      <Tag color="green">{detectionResult.detectionResults.changePercentage.toFixed(2)}%</Tag>
                    </div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>配准置信度:</span>
                      <Tag color={detectionResult.registrationResults.confidenceScore > 0.7 ? 'green' : 'orange'}>
                        {(detectionResult.registrationResults.confidenceScore * 100).toFixed(1)}%
                      </Tag>
                    </div>
                    
                    {detectionResult.detectionResults.processingTime && (
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>处理时间:</span>
                        <Tag>{detectionResult.detectionResults.processingTime.toFixed(1)}秒</Tag>
                      </div>
                    )}

                    <Divider style={{ margin: '8px 0' }} />
                    
                    <div style={{ fontSize: 12, color: '#666' }}>
                      任务ID: {detectionResult.taskId}
                    </div>
                  </Space>
                </Card>
              )}

              {/* 选中的变化区域详情 */}
              {selectedRegion && (
                <Card title="选中区域详情" size="small">
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>变化类型:</span>
                      <Tag color="red">{selectedRegion.changeType}</Tag>
                    </div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>面积:</span>
                      <span>{selectedRegion.area.toFixed(0)} 像素</span>
                    </div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>置信度:</span>
                      <Tag color={selectedRegion.confidence > 0.7 ? 'green' : 'orange'}>
                        {(selectedRegion.confidence * 100).toFixed(1)}%
                      </Tag>
                    </div>
                    
                    <div style={{ fontSize: 12, color: '#666' }}>
                      中心坐标: ({selectedRegion.centroid[0].toFixed(4)}, {selectedRegion.centroid[1].toFixed(4)})
                    </div>
                  </Space>
                </Card>
              )}
            </Space>
          </Col>

          {/* 右侧地图和结果展示 */}
          <Col span={16}>
            <Card 
              title="地图视图" 
              size="small" 
              style={{ height: '100%' }}
              bodyStyle={{ padding: 0, height: 'calc(100% - 57px)' }}
              extra={
                detectionResult && (
                  <Space size="small">
                    <Tag color="blue">
                      {detectionResult.detectionResults.changeRegionsCount} 个变化区域
                    </Tag>
                    <Tag color="green">
                      {detectionResult.detectionResults.changePercentage.toFixed(2)}% 变化
                    </Tag>
                  </Space>
                )
              }
            >
              <MapComponent
                detectionResults={detectionResult ? [detectionResult] : []}
                onRegionSelect={handleRegionSelect}
                showControls={true}
                showLegend={true}
              />
            </Card>
          </Col>
        </Row>
      </Content>
    </Layout>
  );
};

export default DetectionPage;