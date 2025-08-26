import React, { useState, useEffect } from 'react';
import { 
  Layout, 
  Card, 
  Table, 
  Button, 
  Space, 
  Tag, 
  Modal,
  Image,
  Descriptions,
  message,
  Input,
  Select,
  DatePicker,
  Row,
  Col,
  Statistic,
  Empty
} from 'antd';
import { 
  EyeOutlined, 
  DownloadOutlined, 
  SearchOutlined,
  FilterOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import type { ColumnsType, TableProps } from 'antd/es/table';
import dayjs from 'dayjs';
import MapComponent from '@components/Map/MapComponent';
import { getResultApi, handleApiError } from '@services/api';
import type { DetectionResult, ChangeRegion } from '@types/index';

const { Content } = Layout;
const { Search } = Input;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface ResultRecord extends DetectionResult {
  key: string;
}

const ResultsPage: React.FC = () => {
  const [results, setResults] = useState<ResultRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedResult, setSelectedResult] = useState<DetectionResult | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [imagePreviewVisible, setImagePreviewVisible] = useState(false);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  
  // 过滤和搜索状态
  const [searchText, setSearchText] = useState('');
  const [algorithmFilter, setAlgorithmFilter] = useState<string | undefined>(undefined);
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>(null);

  // 模拟数据加载 (实际项目中应该从API获取)
  useEffect(() => {
    loadResults();
  }, []);

  const loadResults = async () => {
    setLoading(true);
    try {
      // 这里应该调用API获取历史结果列表
      // 现在先用模拟数据
      const mockResults: ResultRecord[] = [
        {
          key: '1',
          taskId: 'task_001',
          timestamp: '2024-01-15T10:30:00',
          description: '工业园区建筑物变化检测',
          algorithmType: 'advanced',
          gpsInfo: {
            image1: { latitude: 39.9042, longitude: 116.4074 },
            image2: { latitude: 39.9043, longitude: 116.4075 }
          },
          registrationResults: {
            confidenceScore: 0.89,
            featureMatches: 1245,
            inlierRatio: 0.76,
            registrationError: 2.3,
            transformApplied: true
          },
          detectionResults: {
            changeRegionsCount: 8,
            totalChangeAreaPixels: 45000,
            changePercentage: 3.2,
            processingTime: 28.5
          },
          changeRegions: [
            {
              id: 'region_001',
              area: 15000,
              centroid: [39.9045, 116.4076],
              boundingBox: { x: 100, y: 200, width: 150, height: 100 },
              confidence: 0.92,
              changeType: 'building',
              geoCoordinates: {
                bounds: [[39.9040, 116.4070], [39.9050, 116.4080]]
              }
            }
          ],
          resultImageUrl: '/static/task_001_result.jpg',
          status: 'completed'
        },
        {
          key: '2',
          taskId: 'task_002',
          timestamp: '2024-01-14T15:20:00',
          description: '农田区域植被变化监测',
          algorithmType: 'simple',
          gpsInfo: {
            image1: { latitude: 40.1234, longitude: 116.5678 },
            image2: { latitude: 40.1235, longitude: 116.5679 }
          },
          registrationResults: {
            confidenceScore: 0.67,
            featureMatches: 892,
            inlierRatio: 0.58,
            registrationError: 4.1,
            transformApplied: true
          },
          detectionResults: {
            changeRegionsCount: 3,
            totalChangeAreaPixels: 12500,
            changePercentage: 1.8,
            processingTime: 15.2
          },
          changeRegions: [
            {
              id: 'region_002',
              area: 8000,
              centroid: [40.1236, 116.5680],
              boundingBox: { x: 300, y: 400, width: 120, height: 80 },
              confidence: 0.74,
              changeType: 'vegetation',
              geoCoordinates: {
                bounds: [[40.1230, 116.5675], [40.1240, 116.5685]]
              }
            }
          ],
          resultImageUrl: '/static/task_002_result.jpg',
          status: 'completed'
        }
      ];
      
      setResults(mockResults);
    } catch (error) {
      message.error('加载结果失败: ' + handleApiError(error));
    } finally {
      setLoading(false);
    }
  };

  // 表格列定义
  const columns: ColumnsType<ResultRecord> = [
    {
      title: '任务ID',
      dataIndex: 'taskId',
      key: 'taskId',
      width: 120,
      render: (text) => <code style={{ fontSize: 12 }}>{text}</code>
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
      render: (text) => dayjs(text).format('YYYY-MM-DD HH:mm'),
      sorter: (a, b) => dayjs(a.timestamp).unix() - dayjs(b.timestamp).unix()
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
      render: (text) => text || '无描述'
    },
    {
      title: '算法',
      dataIndex: 'algorithmType',
      key: 'algorithmType',
      width: 80,
      render: (type) => (
        <Tag color={type === 'advanced' ? 'blue' : 'green'}>
          {type === 'advanced' ? '高级' : '简单'}
        </Tag>
      ),
      filters: [
        { text: '高级算法', value: 'advanced' },
        { text: '简单算法', value: 'simple' }
      ],
      onFilter: (value, record) => record.algorithmType === value
    },
    {
      title: '变化区域',
      dataIndex: ['detectionResults', 'changeRegionsCount'],
      key: 'changeRegionsCount',
      width: 90,
      align: 'center',
      sorter: (a, b) => a.detectionResults.changeRegionsCount - b.detectionResults.changeRegionsCount
    },
    {
      title: '变化比例',
      dataIndex: ['detectionResults', 'changePercentage'],
      key: 'changePercentage',
      width: 90,
      align: 'center',
      render: (value) => `${value.toFixed(1)}%`,
      sorter: (a, b) => a.detectionResults.changePercentage - b.detectionResults.changePercentage
    },
    {
      title: '配准置信度',
      dataIndex: ['registrationResults', 'confidenceScore'],
      key: 'confidenceScore',
      width: 110,
      align: 'center',
      render: (value) => (
        <Tag color={value > 0.7 ? 'green' : value > 0.5 ? 'orange' : 'red'}>
          {(value * 100).toFixed(1)}%
        </Tag>
      ),
      sorter: (a, b) => a.registrationResults.confidenceScore - b.registrationResults.confidenceScore
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status) => (
        <Tag color={status === 'completed' ? 'success' : status === 'error' ? 'error' : 'processing'}>
          {status === 'completed' ? '完成' : status === 'error' ? '失败' : '处理中'}
        </Tag>
      )
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_, record) => (
        <Space size="small">
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewDetail(record)}
          >
            查看
          </Button>
          <Button
            size="small"
            icon={<DownloadOutlined />}
            onClick={() => handleDownload(record)}
          >
            下载
          </Button>
        </Space>
      )
    }
  ];

  // 处理查看详情
  const handleViewDetail = (record: ResultRecord) => {
    setSelectedResult(record);
    setDetailModalVisible(true);
  };

  // 处理下载
  const handleDownload = (record: ResultRecord) => {
    // 实现下载功能
    message.info('下载功能开发中...');
  };

  // 处理图像预览
  const handleImagePreview = (url: string) => {
    setImagePreviewUrl(url);
    setImagePreviewVisible(true);
  };

  // 过滤数据
  const filteredResults = results.filter(result => {
    // 文本搜索
    if (searchText && !result.description?.toLowerCase().includes(searchText.toLowerCase()) && 
        !result.taskId.toLowerCase().includes(searchText.toLowerCase())) {
      return false;
    }
    
    // 算法类型过滤
    if (algorithmFilter && result.algorithmType !== algorithmFilter) {
      return false;
    }
    
    // 日期范围过滤
    if (dateRange) {
      const resultDate = dayjs(result.timestamp);
      if (!resultDate.isBetween(dateRange[0], dateRange[1], 'day', '[]')) {
        return false;
      }
    }
    
    return true;
  });

  // 统计信息
  const statistics = {
    total: results.length,
    completed: results.filter(r => r.status === 'completed').length,
    avgChangePercentage: results.length > 0 ? 
      results.reduce((sum, r) => sum + r.detectionResults.changePercentage, 0) / results.length : 0,
    avgProcessingTime: results.length > 0 ? 
      results.reduce((sum, r) => sum + (r.detectionResults.processingTime || 0), 0) / results.length : 0
  };

  return (
    <Layout style={{ height: '100%', background: '#f5f5f5' }}>
      <Content style={{ padding: 16, overflow: 'auto' }}>
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          {/* 统计卡片 */}
          <Row gutter={16}>
            <Col span={6}>
              <Card size="small">
                <Statistic title="总任务数" value={statistics.total} />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic title="已完成" value={statistics.completed} suffix="/ " />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic 
                  title="平均变化比例" 
                  value={statistics.avgChangePercentage} 
                  precision={1}
                  suffix="%" 
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic 
                  title="平均处理时间" 
                  value={statistics.avgProcessingTime} 
                  precision={1}
                  suffix="秒" 
                />
              </Card>
            </Col>
          </Row>

          {/* 搜索和过滤 */}
          <Card size="small">
            <Row gutter={16} align="middle">
              <Col span={6}>
                <Search
                  placeholder="搜索任务ID或描述"
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  style={{ width: '100%' }}
                />
              </Col>
              <Col span={4}>
                <Select
                  placeholder="算法类型"
                  value={algorithmFilter}
                  onChange={setAlgorithmFilter}
                  allowClear
                  style={{ width: '100%' }}
                >
                  <Option value="advanced">高级算法</Option>
                  <Option value="simple">简单算法</Option>
                </Select>
              </Col>
              <Col span={6}>
                <RangePicker
                  placeholder={['开始日期', '结束日期']}
                  value={dateRange}
                  onChange={setDateRange}
                  style={{ width: '100%' }}
                />
              </Col>
              <Col span={8}>
                <Space>
                  <Button icon={<FilterOutlined />}>
                    高级筛选
                  </Button>
                  <Button 
                    icon={<ReloadOutlined />} 
                    onClick={loadResults}
                    loading={loading}
                  >
                    刷新
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>

          {/* 结果表格 */}
          <Card title={`检测结果 (${filteredResults.length})`} size="small">
            {filteredResults.length > 0 ? (
              <Table
                columns={columns}
                dataSource={filteredResults}
                loading={loading}
                size="small"
                scroll={{ x: 1200 }}
                pagination={{
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total, range) => 
                    `第 ${range[0]}-${range[1]} 条，共 ${total} 条记录`,
                }}
              />
            ) : (
              <Empty description="暂无检测结果" />
            )}
          </Card>
        </Space>

        {/* 详情模态框 */}
        <Modal
          title="检测结果详情"
          open={detailModalVisible}
          onCancel={() => setDetailModalVisible(false)}
          width={1200}
          footer={[
            <Button key="close" onClick={() => setDetailModalVisible(false)}>
              关闭
            </Button>,
            <Button 
              key="download" 
              type="primary" 
              icon={<DownloadOutlined />}
              onClick={() => selectedResult && handleDownload(selectedResult)}
            >
              下载报告
            </Button>
          ]}
        >
          {selectedResult && (
            <Row gutter={16}>
              <Col span={12}>
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  {/* 基本信息 */}
                  <Descriptions title="基本信息" bordered size="small" column={1}>
                    <Descriptions.Item label="任务ID">{selectedResult.taskId}</Descriptions.Item>
                    <Descriptions.Item label="创建时间">
                      {dayjs(selectedResult.timestamp).format('YYYY-MM-DD HH:mm:ss')}
                    </Descriptions.Item>
                    <Descriptions.Item label="算法类型">
                      <Tag color={selectedResult.algorithmType === 'advanced' ? 'blue' : 'green'}>
                        {selectedResult.algorithmType === 'advanced' ? '高级算法' : '简单算法'}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="描述">
                      {selectedResult.description || '无描述'}
                    </Descriptions.Item>
                  </Descriptions>

                  {/* 检测结果 */}
                  <Descriptions title="检测结果" bordered size="small" column={1}>
                    <Descriptions.Item label="变化区域数量">
                      {selectedResult.detectionResults.changeRegionsCount}
                    </Descriptions.Item>
                    <Descriptions.Item label="变化面积">
                      {selectedResult.detectionResults.totalChangeAreaPixels} 像素
                    </Descriptions.Item>
                    <Descriptions.Item label="变化比例">
                      {selectedResult.detectionResults.changePercentage.toFixed(2)}%
                    </Descriptions.Item>
                    <Descriptions.Item label="处理时间">
                      {selectedResult.detectionResults.processingTime?.toFixed(1) || '--'} 秒
                    </Descriptions.Item>
                  </Descriptions>

                  {/* 配准信息 */}
                  <Descriptions title="配准信息" bordered size="small" column={1}>
                    <Descriptions.Item label="置信度">
                      <Tag color={selectedResult.registrationResults.confidenceScore > 0.7 ? 'green' : 'orange'}>
                        {(selectedResult.registrationResults.confidenceScore * 100).toFixed(1)}%
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="特征匹配数">
                      {selectedResult.registrationResults.featureMatches}
                    </Descriptions.Item>
                    <Descriptions.Item label="内点比例">
                      {(selectedResult.registrationResults.inlierRatio * 100).toFixed(1)}%
                    </Descriptions.Item>
                    <Descriptions.Item label="配准误差">
                      {selectedResult.registrationResults.registrationError.toFixed(2)} 像素
                    </Descriptions.Item>
                  </Descriptions>

                  {/* 结果图像 */}
                  <Card title="结果图像" size="small">
                    <Image
                      width="100%"
                      src={selectedResult.resultImageUrl}
                      preview={{
                        mask: <EyeOutlined />
                      }}
                      fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1xkE8A="
                    />
                  </Card>
                </Space>
              </Col>
              
              <Col span={12}>
                {/* 地图视图 */}
                <Card title="地图视图" size="small" style={{ height: 600 }}>
                  <MapComponent
                    detectionResults={[selectedResult]}
                    showControls={false}
                    showLegend={true}
                    height={500}
                  />
                </Card>
              </Col>
            </Row>
          )}
        </Modal>

        {/* 图像预览模态框 */}
        <Modal
          open={imagePreviewVisible}
          onCancel={() => setImagePreviewVisible(false)}
          footer={null}
          width="auto"
          style={{ top: 20 }}
        >
          <Image src={imagePreviewUrl} style={{ maxWidth: '100%' }} />
        </Modal>
      </Content>
    </Layout>
  );
};

export default ResultsPage;